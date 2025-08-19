import numpy as np
import os

def generate_dummy_dataset(
    save_dir="./dataset/dummy_eeg_data",
    n_subjects=3,
    n_sessions=3,
    n_trials_per_session=3,
    n_channels=62,
    n_samples_per_slice=250,
    min_slices_per_trial=2,
    max_slices_per_trial=5,
    n_classes=3
):
    """
    生成dummy EEG数据，保存成：
    X.npy              # [total_slices, n_channels, n_sample_per_slice]
    y.npy              # [total_slices]
    subject_ids.npy    # [total_slices]
    trial_ids.npy      # [total_slices]
    session_ids.npy    # [total_slices]
    """
    os.makedirs(save_dir, exist_ok=True)

    X_list = []
    y_list = []
    subject_ids_list = []
    trial_ids_list = []
    session_ids_list = []

    trial_counter = 0  # 全局trial编号

    for subj in range(n_subjects):
        for sess in range(n_sessions):
            for trial in range(n_trials_per_session):
                # 这个 trial 切多少 slice 是随机的（模拟真实情况）
                n_slices = np.random.randint(min_slices_per_trial, max_slices_per_trial + 1)

                # 随机生成 EEG 数据
                trial_data = np.random.randn(n_slices, n_channels, n_samples_per_slice).astype(np.float32)

                # 随机生成标签（一个 trial 内所有 slices 共享同一标签）
                trial_label = np.random.randint(0, n_classes)
                trial_labels = np.full((n_slices,), trial_label, dtype=np.int64)

                # 保存
                X_list.append(trial_data)
                y_list.append(trial_labels)
                subject_ids_list.append(np.full((n_slices,), subj, dtype=np.int64))
                trial_ids_list.append(np.full((n_slices,), trial_counter, dtype=np.int64))
                session_ids_list.append(np.full((n_slices,), sess, dtype=np.int64))

                trial_counter += 1

    # 拼接成统一格式
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(subject_ids_list, axis=0)
    trial_ids = np.concatenate(trial_ids_list, axis=0)
    session_ids = np.concatenate(session_ids_list, axis=0)

    # 保存
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)
    np.save(os.path.join(save_dir, "subject_ids.npy"), subject_ids)
    np.save(os.path.join(save_dir, "trial_ids.npy"), trial_ids)
    np.save(os.path.join(save_dir, "session_ids.npy"), session_ids)

    print(f"Dummy dataset saved in '{save_dir}':")
    print(f"  X.shape = {X.shape}")
    print(f"  y.shape = {y.shape}")
    print(f"  subject_ids.shape = {subject_ids.shape}")
    print(f"  trial_ids.shape = {trial_ids.shape}")
    print(f"  session_ids.shape = {session_ids.shape}")

if __name__ == "__main__":
    generate_dummy_dataset()
