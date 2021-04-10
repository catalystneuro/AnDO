import json
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ando.AnDOChecker import is_valid

REQ_FILES = dict(
    description="dataset_description.json",
    participant="participants.tsv",
    session="{subject_label}\\{subject_label}_sessions.tsv",
    channels="{subject_label}\\{session_label}\\ephys\\{subject_label}_{session_label}_channels.tsv",
    contacts="{subject_label}\\{session_label}\\ephys\\{subject_label}_{session_label}_contacts.tsv",
    ephys="{subject_label}\\{session_label}\\ephys\\{subject_label}_{session_label}_ephys.json",
    probes="{subject_label}\\{session_label}\\ephys\\{subject_label}_{session_label}_probes.tsv",
    file="{subject_label}\\{session_label}\\ephys\\{subject_label}_{session_label}_ephys{file_type}",
)


class BidsConverter(ABC):
    file_type = ""

    def __init__(self, dataset_path, **kwargs):
        self.dataset_path = Path(dataset_path)
        self.output_path = self.dataset_path.parent / "BIDSExt" / self.dataset_path.name
        self._kwargs = kwargs
        self._participants_dict = dict()
        self._dataset_desc_json = dict()
        self._sessions_dict = defaultdict(dict)
        self._channels_dict = defaultdict(dict)
        self._contacts_dict = defaultdict(dict)
        self._ephys_dict = defaultdict(dict)
        self._probes_dict = defaultdict(dict)
        self._file_name_dict = defaultdict(dict)
        self.datafiles_list = list(self.dataset_path.glob(f"**/*{self.file_type}"))
        assert len(self.datafiles_list) > 0, f"no files of type {self.file_type} found"
        self._datafiles_io = self._datafiles_open()
        self._datafiles = [file.read() for file in self._datafiles_io]
        self._labels_dict = defaultdict(list)
        self.extract_metadata()

    @abstractmethod
    def _datafiles_open(self):
        pass

    def __del__(self):
        for file in self._datafiles_io:
            file.close()

    def _tqdm(self, tqdm_obj, desc):
        t = tqdm(tqdm_obj)
        t.set_description(desc)
        return t

    def _create_participant_df(self):
        participants_df = None
        for file in self._tqdm(self._datafiles, "participant"):
            participant_df = self._get_participant_info(file)
            sub_label = participant_df["ParticipantID"][0]
            if sub_label not in self._labels_dict:
                if participants_df is None:
                    participants_df = participant_df
                else:
                    participants_df = participants_df.append(participant_df)
                self._labels_dict[sub_label] = []
        self._participants_dict = self._get_default_dict(
            "participant", data=participants_df
        )[""]

    def _create_description_json(self):
        self._dataset_desc_json = self._get_default_dict(
            "description", data=self._get_dataset_info(self._datafiles[0])
        )[""]

    def _create_session_df(self):
        for file in self._tqdm(self._datafiles, "session"):
            subject_name = self._get_subject_label(file)
            session_label = self._get_session_label(file)
            session_df = self._get_session_info(file)
            sessions_df = self._sessions_dict.get(
                subject_name, dict(data=pd.DataFrame(columns=session_df.columns))
            )["data"]
            if not sessions_df["session_id"].str.contains(session_label).any():
                sessions_df = sessions_df.append(session_df)
                self._labels_dict[subject_name].append(session_label)
            self._sessions_dict[subject_name] = self._get_default_dict(
                "session", subject_name, session_label, data=sessions_df
            )[session_label]

    def _create_sessionlevel_data(self):
        for no, file in enumerate(self._tqdm(self._datafiles, "sessionlevel")):
            subject_name = self._get_subject_label(file)
            session_label = self._get_session_label(file)
            self._probes_dict[subject_name].update(
                self._get_default_dict(
                    "probes",
                    subject_name,
                    session_label,
                    data=self._get_probes_info(file),
                )
            )
            self._ephys_dict[subject_name].update(
                self._get_default_dict(
                    "ephys",
                    subject_name,
                    session_label,
                    data=self._get_ephys_info(file),
                )
            )
            self._channels_dict[subject_name].update(
                self._get_default_dict(
                    "channels",
                    subject_name,
                    session_label,
                    data=self._get_channels_info(file),
                )
            )
            self._contacts_dict[subject_name].update(
                self._get_default_dict(
                    "contacts",
                    subject_name,
                    session_label,
                    data=self._get_contacts_info(file),
                )
            )
            self._file_name_dict[subject_name].update(
                self._get_default_dict(
                    "file", subject_name, session_label, data=self.datafiles_list[no]
                )
            )

    def _get_default_dict(self, key, subject_label="", session_label="", data=None):
        return {
            session_label: dict(
                name=REQ_FILES[key].format(
                    subject_label=subject_label,
                    session_label=session_label,
                    file_type=self.file_type,
                ),
                data=data,
            )
        }

    @abstractmethod
    def _get_subject_label(self, file):
        pass

    @abstractmethod
    def _get_session_label(self, file):
        pass

    @abstractmethod
    def _get_participant_info(self, file):
        pass

    @staticmethod
    @abstractmethod
    def _get_dataset_info(file):
        pass

    @abstractmethod
    def _get_session_info(self, file):
        pass

    @staticmethod
    @abstractmethod
    def _get_contacts_info(file):
        pass

    @staticmethod
    @abstractmethod
    def _get_channels_info(file):
        pass

    @staticmethod
    @abstractmethod
    def _get_ephys_info(file):
        pass

    @staticmethod
    @abstractmethod
    def _get_probes_info(file):
        pass

    def extract_metadata(self):
        self._create_participant_df()
        self._create_session_df()
        self._create_sessionlevel_data()
        self._create_description_json()
        return dict(
            description=self._dataset_desc_json,
            sessions=self._sessions_dict,
            participant=self._participants_dict,
            probes=self._probes_dict,
            ephys=self._ephys_dict,
            channels=self._channels_dict,
            contacts=self._contacts_dict,
            file=self._file_name_dict,
        )

    @staticmethod
    def _write_tsv(data, write_path):
        write_path.parent.mkdir(parents=True, exist_ok=True)
        assert isinstance(data, pd.DataFrame), f"{data} should be a df"
        if not write_path.exists():
            data.dropna(axis="columns", how="all", inplace=True)
            data.to_csv(write_path, sep="\t", index=False)

    @staticmethod
    def _write_json(data, write_path):
        write_path.parent.mkdir(parents=True, exist_ok=True)
        assert isinstance(data, dict), f"{data} should be a dict"
        if not write_path.exists():
            with open(write_path, "w") as j:
                json.dump(data, j)

    @staticmethod
    def _move_data_file(source, dest, move=True):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if move:
            if not dest.exists():
                source.replace(dest)
        else:
            if not dest.exists():
                dest.symlink_to(source)

    def organize(self, output_path=None, move_file=False, re_write=True, validate=True):
        if output_path is not None:
            self.output_path = Path(output_path)
        if re_write and self.output_path.exists():
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        data, loc = self._parse_data_dict(self._participants_dict, self.output_path)
        self._write_tsv(data, loc)
        data, loc = self._parse_data_dict(self._dataset_desc_json, self.output_path)
        self._write_json(data, loc)
        for participant, sessions in self._labels_dict.items():
            data, loc = self._parse_data_dict(
                self._sessions_dict[participant], self.output_path
            )
            self._write_tsv(data, loc)
            for session in sessions:
                data, loc = self._parse_data_dict(
                    self._channels_dict[participant][session], self.output_path
                )
                self._write_tsv(data, loc)
                data, loc = self._parse_data_dict(
                    self._probes_dict[participant][session], self.output_path
                )
                self._write_tsv(data, loc)
                data, loc = self._parse_data_dict(
                    self._contacts_dict[participant][session], self.output_path
                )
                self._write_tsv(data, loc)
                data, loc = self._parse_data_dict(
                    self._ephys_dict[participant][session], self.output_path
                )
                self._write_json(data, loc)
                data, loc = self._parse_data_dict(
                    self._file_name_dict[participant][session], self.output_path
                )
                self._move_data_file(data, loc, move=move_file)
        if validate:
            is_valid(self.output_path)

    def _parse_data_dict(self, data_dict, output_path):
        return data_dict["data"], output_path / data_dict["name"]

    def get_subject_names(self):
        return list(self._participants_dict["data"]["ParticipantID"])

    def get_session_names(self, subject_name=None):
        if subject_name is None:
            subject_name = self.get_subject_names()[0]
        return list(self._sessions_dict[subject_name]["data"]["session_id"])

    def get_channels_info(self, subject_name=None, session_name=None):
        if subject_name is None:
            subject_name = self.get_subject_names()[0]
        if session_name is None:
            session_name = self.get_session_names()[0]
        return self._channels_dict[subject_name][session_name]["data"].to_dict()

    def get_contacts_info(self, subject_name=None, session_name=None):
        if subject_name is None:
            subject_name = self.get_subject_names()[0]
        if session_name is None:
            session_name = self.get_session_names()[0]
        return self._contacts_dict[subject_name][session_name]["data"].to_dict()

    def get_ephys_info(self, subject_name=None, session_name=None):
        if subject_name is None:
            subject_name = self.get_subject_names()[0]
        if session_name is None:
            session_name = self.get_session_names()[0]
        return self._ephys_dict[subject_name][session_name]["data"]

    def get_probes_info(self, subject_name=None, session_name=None):
        if subject_name is None:
            subject_name = self.get_subject_names()[0]
        if session_name is None:
            session_name = self.get_session_names()[0]
        return self._probes_dict[subject_name][session_name]["data"].to_dict()

    def get_participants_info(self):
        return self._participants_dict["data"].to_dict()

    def get_dataset_description(self):
        return self._dataset_desc_json["data"]

    def get_session_info(self, subject_name=None):
        if subject_name is None:
            subject_name = self.get_subject_names()[0]
        return self._sessions_dict[subject_name]["data"].to_dict()
