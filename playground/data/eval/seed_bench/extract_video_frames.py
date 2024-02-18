import os
import json
import numpy as np
import torch
import av
from decord import VideoReader, cpu
from PIL import Image
import random

from tqdm.auto import tqdm
import concurrent.futures


num_segments = 1

# root directory of evaluation dimension 10
dimension10_dir = "./videos/20bn-something-something-v2"
# root directory of evaluation dimension 11
dimension11_dir = "./videos/EPIC-KITCHENS"
# root directory of evaluation dimension 12
dimension12_dir = "./videos/BreakfastII_15fps_qvga_sync"

def transform_video(buffer):
    try:
        buffer = buffer.numpy()
    except AttributeError:
        try:
            buffer = buffer.asnumpy()
        except AttributeError:
            print("Both buffer.numpy() and buffer.asnumpy() failed.")
            buffer = None
    images_group = list()
    for fid in range(len(buffer)):
        images_group.append(Image.fromarray(buffer[fid]))
    return images_group

def get_index(num_frames, num_segments):
    if num_segments > num_frames:
        offsets = np.array([
            idx for idx in range(num_frames)
        ])
    else:
        # uniform sampling
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
    return offsets


def fetch_images(qa_item):
    use_pyav = False
    segment = None
    if qa_item['question_type_id'] == 10:
        data_path = os.path.join(dimension10_dir, qa_item['data_id'])
        start = 0.0
        end = 0.0
    elif qa_item['question_type_id'] == 11:
        data_path = os.path.join(dimension11_dir, qa_item['data_id'].split('/')[-1])
        segment = qa_item['segment']
        start, end = segment[0], segment[1]
    elif qa_item['question_type_id'] == 12:
        data_path = os.path.join(dimension12_dir, qa_item['data_id'])
        segment = qa_item['segment']
        start, end = segment[0], segment[1]
        use_pyav = True

    if use_pyav:
        # using pyav for decoding videos in evaluation dimension 12
        reader = av.open(data_path)
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
        video_len = len(frames)
        start_frame, end_frame = start, end
        end_frame = min(end_frame, video_len)
        offset = get_index(end_frame - start_frame, num_segments)
        frame_indices = offset + start_frame
        buffer = torch.stack([frames[idx] for idx in frame_indices])
    else:
        # using decord for decoding videos in evaluation dimension 10-11
        vr = VideoReader(data_path, num_threads=1, ctx=cpu(0))
        video_len = len(vr)
        fps = vr.get_avg_fps()
        if segment is not None:
            # obtain start and end frame for the video segment in evaluation dimension 11
            start_frame = int(min(max(start * fps, 0), video_len - 1))
            end_frame = int(min(max(end * fps, 0), video_len - 1))
            tot_frames = int(end_frame - start_frame)
            offset = get_index(tot_frames, num_segments)
            frame_indices = offset + start_frame
        else:
            # sample frames of the video in evaluation dimension 10
            frame_indices = get_index(video_len - 1, num_segments)
        vr.seek(0)
        buffer = vr.get_batch(frame_indices)
    return transform_video(buffer)


def fetch_images_parallel(qa_item):
    return qa_item, fetch_images(qa_item)

if __name__ == "__main__":
    data = json.load(open('SEED-Bench.json'))
    video_img_dir = 'SEED-Bench-video-image'
    ques_type_id_to_name = {id:n for n,id in data['question_type'].items()}

    video_data = [x for x in data['questions'] if x['data_type'] == 'video']

    with open(output, 'w') as f, concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_images = {executor.submit(fetch_images_parallel, qa_item): qa_item for qa_item in video_data}
        for future in tqdm(concurrent.futures.as_completed(future_to_images), total=len(future_to_images)):
            qa_item = future_to_images[future]
            try:
                qa_item, images = future.result()
            except Exception as exc:
                print(f'{qa_item} generated an exception: {exc}')
            else:
                img_file = f"{qa_item['question_type_id']}_{qa_item['question_id']}.png"
                images[0].save(os.path.join(video_img_dir, img_file))
