import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
import torchvision.transforms as T
import re
import os

class CLEVRERQADataset(Dataset):
    """
    Standardized Dataset for all CLEVRER tasks.
    Always returns: (video, label, task_type, video_idx, question_text, choices_text)
    """
    def __init__(self, split='train', num_frames=16, frame_size=112, task_type='descriptive', tensor_dir='data/clevrer_tensors'):
        super().__init__()
        self.task_type = task_type
        self.split = split
        self.qa_data = load_dataset('zechen-nlp/clevrer', task_type, split=split)
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        if tensor_dir == 'data/clevrer_tensors':
            if split == 'validation': self.tensor_dir = 'data/clevrer_tensors_val'
            elif split == 'test': self.tensor_dir = 'data/clevrer_tensors_test'
            else: self.tensor_dir = tensor_dir
        else: self.tensor_dir = tensor_dir
            
        self.resize = T.Resize((frame_size, frame_size), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.answer_to_idx = {
            'cube': 0, 'cylinder': 1, 'sphere': 2,
            'gray': 3, 'red': 4, 'blue': 5, 'green': 6, 'brown': 7, 'purple': 8, 'cyan': 9, 'yellow': 10,
            'metal': 11, 'rubber': 12, 'yes': 13, 'no': 14
        }
        for i in range(11): self.answer_to_idx[str(i)] = 15 + i

    def __len__(self): return len(self.qa_data)

    def _extract_video_index(self, video_path):
        match = re.search(r'video_(\d+)\.mp4', video_path)
        return int(match.group(1)) if match else 0

    def __getitem__(self, idx):
        qa_item = self.qa_data[idx]
        choices_text = ["None"] * 4 # Use "None" string to prevent RNN length 0 errors
        
        if self.task_type == 'descriptive':
            question_str = qa_item['conversations']['value'][0].lower()
            answer_str = qa_item['conversations']['value'][1].lower()
            label = torch.tensor(self.answer_to_idx.get(answer_str, 0))
        else:
            question_str = qa_item['question'].lower()
            raw_choices = qa_item['choices']['choice']
            for i in range(min(len(raw_choices), 4)):
                choices_text[i] = raw_choices[i].lower()
            ans_labels = qa_item['choices']['answer']
            label = torch.tensor([1.0 if c == 'correct' else 0.0 for c in ans_labels])
            if label.shape[0] < 4: label = torch.cat([label, torch.zeros(4 - label.shape[0])])
            elif label.shape[0] > 4: label = label[:4]

        video_idx = self._extract_video_index(qa_item['video'])
        fast_path = os.path.join(self.tensor_dir, f"video_{video_idx:05d}.pth")
        
        if os.path.exists(fast_path):
            try:
                full_video = torch.load(fast_path, map_location='cpu', weights_only=False)
                total_frames = full_video.shape[1]
                indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long().tolist()
                clip = full_video[:, indices].float() / 255.0
                if full_video.shape[-1] != self.frame_size: clip = self.resize(clip)
            except Exception: clip = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
        else: clip = torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)

        processed_frames = [self.normalize(clip[:, t]) for t in range(self.num_frames)]
        clip_final = torch.stack(processed_frames, dim=1) 

        return clip_final, label, self.task_type, video_idx, question_str, choices_text

def get_clevrer_qa_loaders(split='train', batch_size=32, num_frames=16, frame_size=112):
    tasks = ['descriptive', 'explanatory', 'predictive', 'counterfactual']
    datasets = [CLEVRERQADataset(split=split, num_frames=num_frames, frame_size=frame_size, task_type=t) for t in tasks]
    return ConcatDataset(datasets)
