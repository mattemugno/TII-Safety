import os

from utils import read_json


def frame_to_text(sequence_id, frame, sequence):
    timestamp = frame.split('_')[0]
    keypoints = sequence[frame]['keypoints']
    gloves = sequence[frame]['gloves_flag']
    occlusion = sequence[frame]['occlusion_flags']

    return (
        f"Sequence ID: {sequence_id}, Timestamp: {timestamp}, "
        f"LH: {keypoints['lh'][:3]}, RH: {keypoints['rh'][:3]}, "
        f"LE: {keypoints['le'][:3]}, RE: {keypoints['re'][:3]}, "
        f"LS: {keypoints['ls'][:3]}, RS: {keypoints['rs'][:3]}, "
        f"Gloves: LH={gloves['g_lh']}, RH={gloves['g_rh']}, "
        f"Occlusion: LH={occlusion['o_lh']}, RH={occlusion['o_rh']}"
    )


def convert_sequence_to_text(sequence, seq_id):
    return "\n".join(frame_to_text(seq_id, frame, sequence) for frame in sequence)


def convert_json_to_text(file_path):
    data = read_json(file_path)

    return "\n\n".join(
        convert_sequence_to_text(sequence_data, seq_id)
        for seq_id, sequence_data in data.items()
    )


dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
subjects = ["francesco", "matteo", "michele", "michela"]

for subject in subjects:
    subject_data_path = os.path.join(dataset_path, subject + " gloves")

    data_path = os.path.join(subject_data_path, f'keypoints {subject}.json')

    text_output = convert_json_to_text(data_path)

    output_file_path = os.path.join(subject_data_path, f'keypoints_{subject}_text_output.txt')

    with open(output_file_path, 'w') as f:
        f.write(text_output)

    print(f"Conversion complete! Text has been saved at: {output_file_path}")
