# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

from .packing import EpisodePackingInfo, PackedSampleRef, round_up

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    if getattr(config, "dataset_type", None) == "ebench":
        from .ebench_latent_dataset import EBenchLatentDataset

        return EBenchLatentDataset(
            repo_id=repo_id,
            config=config,
        )
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = sorted(v.split('/meta/info.json')[0] for v in repo_list)
    if not repo_list:
        raise FileNotFoundError(f"No LeRobot datasets found under {config.dataset_path}")
    worker_count = min(max(int(num_init_worker), 1), len(repo_list))
    if worker_count == 1:
        datasets_out_lst = [construct_func(repo_id) for repo_id in repo_list]
    else:
        with Pool(worker_count) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self.config = config
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )
        self._text_to_id = None
        self._packing_manifest = None
        stats_paths = {
            Path(dataset.action_stats_path).resolve()
            for dataset in self._datasets
            if hasattr(dataset, "action_stats_path")
        }
        stats_hashes = {
            dataset.action_stats_sha256
            for dataset in self._datasets
            if hasattr(dataset, "action_stats_sha256")
        }
        if len(stats_paths) > 1 or len(stats_hashes) > 1:
            raise ValueError("Child datasets use different EBench action stats artifacts")
        self.action_stats_path = next(iter(stats_paths), None)
        self.action_stats_sha256 = next(iter(stats_hashes), None)
        segment_paths = {
            Path(dataset.segment_manifest_path).resolve()
            for dataset in self._datasets
            if getattr(dataset, "segment_manifest_path", None) is not None
        }
        segment_hashes = {
            dataset.segment_manifest_sha256
            for dataset in self._datasets
            if getattr(dataset, "segment_manifest_path", None) is not None
        }
        segment_audit_paths = {
            Path(dataset.segment_audit_path).resolve()
            for dataset in self._datasets
            if getattr(dataset, "segment_audit_path", None) is not None
        }
        if any(
            len(values) > 1
            for values in (segment_paths, segment_hashes, segment_audit_paths)
        ):
            raise ValueError("Child datasets use different EBench segment artifacts")
        self.segment_manifest_path = next(iter(segment_paths), None)
        self.segment_manifest_sha256 = next(iter(segment_hashes), None)
        self.segment_audit_path = next(iter(segment_audit_paths), None)
        text_paths = {
            Path(dataset.text_embeddings_path).resolve()
            for dataset in self._datasets
            if hasattr(dataset, "text_embeddings_path")
        }
        text_hashes = {
            dataset.text_embeddings_sha256
            for dataset in self._datasets
            if hasattr(dataset, "text_embeddings_sha256")
        }
        empty_paths = {
            Path(dataset.empty_embedding_path).resolve()
            for dataset in self._datasets
            if hasattr(dataset, "empty_embedding_path")
        }
        empty_hashes = {
            dataset.empty_embedding_sha256
            for dataset in self._datasets
            if hasattr(dataset, "empty_embedding_sha256")
        }
        if any(
            len(values) > 1
            for values in (text_paths, text_hashes, empty_paths, empty_hashes)
        ):
            raise ValueError("Child datasets use different audited text artifacts")
        self.text_embeddings_path = next(iter(text_paths), None)
        self.text_embeddings_sha256 = next(iter(text_hashes), None)
        self.empty_embedding_path = next(iter(empty_paths), None)
        self.empty_embedding_sha256 = next(iter(empty_hashes), None)

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def set_text_lookup(self, text_keys):
        self._text_to_id = {text: idx for idx, text in enumerate(text_keys)}

    def __getitem__(self, idx) -> dict:
        packed_ref = idx if isinstance(idx, PackedSampleRef) else None
        item_idx = packed_ref.index if packed_ref is not None else int(idx)
        assert 0 <= item_idx < len(self)
        dataset_id = self.item_id_to_dataset_id[item_idx]
        cur_dset = self._datasets[dataset_id]
        local_idx = item_idx - self.acc_dset_num[dataset_id]
        sample = cur_dset[local_idx]
        lineage = sample.get("lineage")
        if lineage is not None:
            lineage.update(
                dataset_index=int(item_idx),
                local_dataset_index=int(local_idx),
                child_dataset_id=int(dataset_id),
            )
        if self._text_to_id is not None:
            text = sample.pop("text")
            try:
                sample["text_id"] = self._text_to_id[text]
            except KeyError as error:
                raise KeyError(f"Text {text!r} is missing from the shared cache") from error
        if packed_ref is not None:
            sample.update(
                sample_uid=packed_ref.sample_uid,
                update_id=packed_ref.update_id,
                micro_idx=packed_ref.micro_idx,
                micro_count=packed_ref.micro_count,
                planned_tokens=packed_ref.planned_tokens,
            )
            lineage = sample.get("lineage")
            if lineage is not None:
                lineage.update(
                    sample_uid=int(packed_ref.sample_uid),
                    update_id=int(packed_ref.update_id),
                    micro_idx=int(packed_ref.micro_idx),
                    micro_count=int(packed_ref.micro_count),
                    planned_tokens=int(packed_ref.planned_tokens),
                )
        return sample

    def get_packing_manifest(self):
        if self._packing_manifest is not None:
            return self._packing_manifest
        manifest = []
        global_idx = 0
        for dataset in self._datasets:
            for local_idx in range(len(dataset)):
                metadata = dataset.get_packing_metadata(local_idx)
                manifest.append(EpisodePackingInfo(index=global_idx, **metadata))
                global_idx += 1
        self._packing_manifest = tuple(manifest)
        return self._packing_manifest

    def get_lineage_metadata(self, idx):
        assert 0 <= int(idx) < len(self)
        dataset_id = self.item_id_to_dataset_id[int(idx)]
        cur_dset = self._datasets[dataset_id]
        local_idx = int(idx) - self.acc_dset_num[dataset_id]
        metadata = cur_dset.get_lineage_metadata(local_idx)
        metadata.update(
            dataset_index=int(idx),
            local_dataset_index=int(local_idx),
            child_dataset_id=int(dataset_id),
        )
        return metadata

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.config = config
        self.defer_text_embedding = bool(
            getattr(config, "defer_text_embedding", False)
        )
        self.empty_emb = None
        if not self.defer_text_embedding:
            self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        episode_values = sorted(
            self.meta.episodes.values(), key=lambda value: int(value["episode_index"])
        )
        for value in episode_values:
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = sorted(
                value["action_config"],
                key=lambda item: (int(item["start_frame"]), int(item["end_frame"])),
            )
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False, mmap=True)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        out_dict = dict(latents=cat_latent)
        if self.defer_text_embedding:
            out_dict["text"] = str(data_dict[f"{self.used_video_keys[0]}.text"])
        else:
            text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
            if (
                not getattr(self.config, "defer_cfg_dropout", False)
                and torch.rand(1).item() < self.cfg_prob
            ):
                text_emb = self.empty_emb
            out_dict["text_emb"] = text_emb
        return out_dict

    def _latent_file_path(self, meta, camera_key):
        episode_index = meta["episode_index"]
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return (
            Path(self.latent_path)
            / f"chunk-{episode_chunk:03d}"
            / camera_key
            / (
                f"episode_{episode_index:06d}_{meta['start_frame']}_"
                f"{meta['end_frame']}.pth"
            )
        )

    def get_packing_metadata(self, idx):
        """Read scalar tensor metadata without paging repeated text embeddings."""
        meta = self.new_metas[idx]
        records = [
            torch.load(
                self._latent_file_path(meta, key),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            for key in self.used_video_keys
        ]
        frame_counts = {int(record["latent_num_frames"]) for record in records}
        channel_counts = {int(record["latent"].shape[-1]) for record in records}
        if len(frame_counts) != 1 or len(channel_counts) != 1:
            raise ValueError(f"Camera latent mismatch for {self.repo_id}, sample {idx}")
        frame_count = frame_counts.pop()
        channels = channel_counts.pop()
        frame_ids = records[0]["frame_ids"]
        if len(frame_ids) < 2:
            raise ValueError(f"Not enough frame ids for {self.repo_id}, sample {idx}")
        action_tokens_per_frame = int(frame_ids[1] - frame_ids[0]) * 4
        if action_tokens_per_frame != int(self.config.action_per_frame):
            raise ValueError(
                f"Action tokens/frame mismatch for {self.repo_id}, sample {idx}: "
                f"data={action_tokens_per_frame}, config={self.config.action_per_frame}"
            )
        heights = [int(record["latent_height"]) for record in records]
        widths = [int(record["latent_width"]) for record in records]
        if self.config.env_type == "robotwin_tshape":
            wrist_width = sum(widths[1:])
            if wrist_width != widths[0] or len(set(heights[1:])) != 1:
                raise ValueError(
                    f"Robotwin camera layout is incompatible for {self.repo_id}, "
                    f"sample {idx}"
                )
            video_height = heights[0] + heights[1]
            video_width = widths[0]
        else:
            if len(set(heights)) != 1:
                raise ValueError(f"Camera heights differ for {self.repo_id}, sample {idx}")
            video_height = heights[0]
            video_width = sum(widths)

        patch_f, patch_h, patch_w = self.config.patch_size
        if patch_f != 1:
            raise ValueError("True episode packing currently requires patch_size[0] == 1")
        if video_height % patch_h or video_width % patch_w:
            raise ValueError(
                f"Latent spatial shape {(video_height, video_width)} is not divisible "
                f"by patch {(patch_h, patch_w)}"
            )
        video_tokens = frame_count * (video_height // patch_h) * (
            video_width // patch_w
        )
        action_tokens = frame_count * action_tokens_per_frame
        token_count = round_up(2 * (video_tokens + action_tokens), 128)
        sample_key = "|".join(
            str(value)
            for value in (
                Path(self.repo_id).resolve(),
                meta["episode_index"],
                meta["start_frame"],
                meta["end_frame"],
            )
        )
        return {
            "sample_key": sample_key,
            "frame_count": frame_count,
            "token_count": token_count,
            "shape_signature": (
                channels,
                video_height,
                video_width,
                int(self.config.action_dim),
                action_tokens_per_frame,
                1,
            ),
        }

    def _lineage_metadata_from_frame_ids(self, idx, latent_frame_ids):
        cur_meta = self.new_metas[int(idx) % len(self.new_metas)]
        episode_index = int(cur_meta["episode_index"])
        local_start_frame = int(cur_meta["start_frame"])
        local_end_frame = int(cur_meta["end_frame"])
        frame_ids = np.asarray(latent_frame_ids, dtype=np.int64)
        if frame_ids.ndim != 1 or frame_ids.size < 2:
            raise ValueError(
                f"Malformed frame_ids for {self.repo_id}, sample {idx}: "
                f"shape={frame_ids.shape}"
            )
        frame_stride = int(frame_ids[1] - frame_ids[0])
        if frame_stride <= 0:
            raise ValueError(
                f"Non-positive frame stride for {self.repo_id}, sample {idx}: "
                f"{frame_stride}"
            )
        latent_frame_num = (len(frame_ids) - 1) // 4 + 1
        action_tokens_per_latent = frame_stride * 4
        observation_timestamps = [
            int(frame_ids[min(i * 4, len(frame_ids) - 1)])
            for i in range(latent_frame_num)
        ]
        act_shift = int(frame_ids[0] - local_start_frame)
        prefix_count = action_tokens_per_latent
        required_action_num = latent_frame_num * action_tokens_per_latent
        target_timestamps = [-1] * min(prefix_count, required_action_num)
        real_start = local_start_frame + act_shift
        real_available = max(local_end_frame - real_start, 0)
        real_needed = required_action_num - len(target_timestamps)
        target_timestamps.extend(
            range(real_start, real_start + min(real_available, real_needed))
        )
        if len(target_timestamps) < required_action_num:
            target_timestamps.extend([-1] * (required_action_num - len(target_timestamps)))
        task = str(cur_meta.get("action_text") or cur_meta["tasks"][0])
        return {
            "dataset": Path(self.repo_id).name,
            "dataset_root": str(Path(self.repo_id).resolve()),
            "task": task,
            "episode": episode_index,
            "dataset_index": None,
            "local_dataset_index": int(idx),
            "requested_anchor_t": local_start_frame,
            "remapped_anchor_t": local_start_frame,
            "segment_start_t": local_start_frame,
            "segment_end_t": local_end_frame,
            "observation_timestamps": observation_timestamps,
            "action_target_timestamps": [int(value) for value in target_timestamps],
            "action_tokens_per_latent": int(action_tokens_per_latent),
            "padding_target_count": int(sum(value < 0 for value in target_timestamps)),
            "real_target_count": int(sum(value >= 0 for value in target_timestamps)),
            "timestamp_unit": "raw_episode_step",
            "episode_time_origin": 0,
        }

    def get_lineage_metadata(self, idx):
        meta = self.new_metas[int(idx) % len(self.new_metas)]
        latent_file = self._latent_file_path(meta, self.used_video_keys[0])
        latent_data = torch.load(
            latent_file,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        return self._lineage_metadata_from_frame_ids(idx, latent_data["frame_ids"])
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])
        out_dict["lineage"] = self._lineage_metadata_from_frame_ids(
            idx, latent_frame_ids
        )

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
