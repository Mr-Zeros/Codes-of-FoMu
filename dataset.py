from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, csv_file, wsi_dir, rad_dir, save_dir='saved_features'):
        """
        Args:
            csv_file (string): Path to the CSV file containing case information.
            wsi_dir (string): Root directory containing WSI features.
            rad_dir (string): Root directory containing MRI features.
            save_dir (string): Directory to save or load precomputed features.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.wsi_dir = wsi_dir
        self.rad_dir = rad_dir
        self.save_dir = save_dir

        # Attempt to load pre-saved features
        self.wsi_features = self.load_saved_features('wsi_features_dict.pt')
        self.rad_features = self.load_saved_features('mri_features_dict.pt')
        self.clinical_features = self.load_clinical_features()

    def load_saved_features(self, filename):
        """
        Attempts to load pre-saved feature files.
        Args:
            filename (string): Name of the saved feature file.
        Returns:
            torch.Tensor or None: Loaded features if available, otherwise None.
        """
        file_path = os.path.join(self.save_dir, filename)
        if os.path.exists(file_path):
            print(f"Loading {filename} from {file_path}")
            return torch.load(file_path)
        return None

    def save_features(self, features, filename):
        """
        Saves the features to a file.
        Args:
            features (torch.Tensor): The features to save.
            filename (string): Name of the file to save features.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        file_path = os.path.join(self.save_dir, filename)
        torch.save(features, file_path)
        print(f"Saved {filename} to {file_path}")

    def load_WSI_features(self):
        """
        Preloads all WSI features.
        Returns:
            dict: A dictionary with case IDs as keys and corresponding WSI features as values.
        """
        features_dict = {}
        for idx in tqdm(range(len(self.data_frame)), desc="Preloading WSI features"):
            case_id = str(int(self.data_frame.iloc[idx]['case_id']))
            wsi_marker = int(self.data_frame.iloc[idx]['WSI_marker'])
            if wsi_marker == 1:  # Only load features if WSI_marker is 1
                matched_slide_paths = [
                    os.path.join(self.wsi_dir, d) for d in os.listdir(self.wsi_dir) if d.startswith(case_id)
                ]
                path_features = []
                for slide_path in matched_slide_paths:
                    wsi_feat_path = os.path.join(slide_path, 'features.pt')
                    if os.path.exists(wsi_feat_path):
                        wsi_bag = torch.load(wsi_feat_path)
                        path_features.append(wsi_bag)
                if path_features:  # Add features only if available
                    features_dict[case_id] = torch.cat(path_features, dim=0)
        self.save_features(features_dict, 'wsi_features_dict.pt')
        return features_dict

    def load_RAD_features(self):
        """
        Preloads all MRI features.
        Returns:
            dict: A dictionary with case IDs as keys and corresponding MRI features as values.
        """
        features_dict = {}
        for idx in tqdm(range(len(self.data_frame)), desc="Preloading MRI features"):
            case_id = str(int(self.data_frame.iloc[idx]['case_id']))
            mri_marker = int(self.data_frame.iloc[idx]['MRI_marker'])
            if mri_marker == 1:  # Only load features if MRI_marker is 1
                modality_features = []
                modalities = ['ADC', 'CE', 'T2', 'DWI']
                for modality in modalities:
                    mri_feat_path = os.path.join(self.rad_dir, f'feat_{case_id}{modality}.pth')
                    if os.path.exists(mri_feat_path):
                        modality_features.append(torch.load(mri_feat_path))
                if modality_features:  # Add features only if available
                    features_dict[case_id] = torch.stack(modality_features, dim=0)
        self.save_features(features_dict, 'mri_features_dict.pt')
        return features_dict

    def load_clinical_features(self):
        """
        Preloads all clinical data, such as Platinum resistance, PARPi, and FIGO stage.
        Returns:
            dict: A dictionary with case IDs as keys and corresponding clinical data as values.
        """
        features_dict = {}
        for idx in range(len(self.data_frame)):
            case_id = str(int(self.data_frame.iloc[idx]['case_id']))
            platinum = self.data_frame.iloc[idx]['Platinum resistance']
            parpi = self.data_frame.iloc[idx]['PARPi']
            figo_stage = self.data_frame.iloc[idx]['FIGO stage']
            clinical_data = torch.tensor([platinum, parpi, figo_stage], dtype=torch.float32)
            features_dict[case_id] = clinical_data
        return features_dict

    def __len__(self):
        """
        Returns the number of cases in the dataset.
        """
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Retrieves data for a specific index from the dataset.
        Args:
            idx (int): Index of the data to retrieve.
        Returns:
            tuple: Contains case_id, WSI features, MRI features, clinical features, label, event time, and censorship.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract relevant information from DataFrame
        case_id, label, event_time, censor = self.data_frame.iloc[idx, [self.data_frame.columns.get_loc(col) for col in ['case_id', 'disc_label', 'survival_months', 'censorship']]]
        case_id = str(int(case_id))

        # Retrieve preloaded features
        data_WSI = self.wsi_features.get(case_id, torch.Tensor())
        data_RAD = self.rad_features.get(case_id, torch.Tensor())
        data_clinical = self.clinical_features.get(case_id, torch.Tensor())

        return case_id, data_WSI, data_RAD, data_clinical, label, event_time, censor
