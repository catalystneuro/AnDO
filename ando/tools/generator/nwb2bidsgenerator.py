import re

import pandas as pd
from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries

from .bidsconverter import BidsConverter


class NwbToBIDS(BidsConverter):
    file_type = ".nwb"

    def __init__(self, dataset_path, **kwargs):
        super().__init__(dataset_path, **kwargs)

    def _datafiles_open(self):
        return [
            NWBHDF5IO(str(file), "r")
            for file in self._tqdm(self.datafiles_list, "reading nwbfiles")
        ]

    def _get_subject_label(self, file, subject_suffix=""):
        if file.subject is not None:
            sb = file.subject
            if sb.subject_id is not None:
                sub_id = re.sub(r"[\W_]+", "", sb.subject_id)
                subject_label = f"sub-{sub_id}"
            else:
                subject_label = f'sub-{sb.date_of_birth.strftime("%Y%m%dT%H%M")}'
        else:
            subject_label = f"sub-noname{subject_suffix}"
        return subject_label

    def _get_session_label(self, file):
        if file.session_id is not None:
            ses_id = re.sub(r"[\W_]+", "", file.session_id)
            session_label = f"ses-{ses_id}"
        else:
            session_label = f'ses-{file.session_start_time.strftime("%Y%m%dT%H%M")}'
        return session_label

    def _get_participant_info(self, nwbfile, subject_suffix=""):
        subject_label = self._get_subject_label(nwbfile, subject_suffix)
        participant_df = pd.DataFrame(
            columns=[
                "Species",
                "ParticipantID",
                "Sex",
                "Birthdate",
                "Age",
                "Genotype",
                "Weight",
            ]
        )
        if nwbfile.subject is not None:
            sb = nwbfile.subject
            df_row = [
                sb.species,
                subject_label,
                sb.sex[0] if sb.sex is not None else None,
                sb.date_of_birth,
                sb.age,
                sb.genotype,
                sb.weight,
            ]
        else:
            df_row = [None, subject_label, None, None, None, None, None]
        participant_df.loc[0] = df_row
        return participant_df

    @staticmethod
    def _get_dataset_info(nwbfile):
        return dict(
            InstitutionName=nwbfile.institution,
            InstitutionalDepartmentName=nwbfile.lab,
            Name="Electrophysiology",
            BIDSVersion="1.0.X",
            Licence="CC BY 4.0",
            Authors=[
                list(nwbfile.experimenter) if nwbfile.experimenter is not None else None
            ][0],
        )

    def _get_session_info(self, nwbfile):
        trials_len = len(nwbfile.trials) if nwbfile.trials is not None else None
        session_label = self._get_session_label(nwbfile)
        session_df = pd.DataFrame(
            columns=["session_id", "#_trials", "comment"],
            data=[[session_label, trials_len, nwbfile.session_description]],
        )
        return session_df

    @staticmethod
    def _get_channels_info(nwbfile):
        channels_df = pd.DataFrame(
            columns=[
                "channel_id",
                "Contact_id",
                "type",
                "units",
                "sampling_frequency",
                "unit_conversion_multiplier",
            ]
        )
        es = [i for i in nwbfile.children if isinstance(i, ElectricalSeries)]
        if len(es) > 0:
            es = es[0]
            no_channels = es.data.shape[1]
            sampling_frequency = es.rate
            conversion = es.conversion
            unit = es.unit
            for chan_no in range(no_channels):
                channels_df.loc[len(channels_df.index)] = [
                    chan_no,
                    chan_no,
                    "neural signal",
                    unit,
                    sampling_frequency,
                    conversion,
                ]
        return channels_df

    @staticmethod
    def _get_ephys_info(nwbfile, **kwargs):
        return dict(PowerLineFrequency=kwargs.get("powerlinefrequency", 50.0))

    @staticmethod
    def _get_contacts_info(nwbfile):
        contacts_df = pd.DataFrame(
            columns=["x", "y", "z", "impedance", "contact_id", "probe_id", "Location"]
        )
        e_table = nwbfile.electrodes
        if e_table is not None:
            for contact_no in range(len(e_table)):
                contacts_df.loc[len(contacts_df.index)] = [
                    e_table.x[contact_no],
                    e_table.y[contact_no],
                    e_table.z[contact_no],
                    e_table.imp[contact_no],
                    contact_no,
                    e_table.group[contact_no].device.name,
                    e_table.location[contact_no],
                ]
        return contacts_df

    def _get_probes_info(self, nwbfile, **kwargs):
        contacts_df = self._get_contacts_info(nwbfile)
        probes_df = pd.DataFrame(columns=["probeID", "type"])
        for probe_id in contacts_df["probe_id"].unique():
            probes_df.loc[len(probes_df.index)] = [
                probe_id,
                kwargs.get("probe_type", "acute"),
            ]
        return probes_df
