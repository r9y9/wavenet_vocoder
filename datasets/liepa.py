from __future__ import with_statement, print_function, absolute_import

from nnmnkwii.datasets import FileDataSource

import numpy as np
from os.path import join, splitext, isdir
from os import listdir

# List of available speakers.
available_speakers = [
    'D587', 'D175', 'D114', 'D127', 'D172', 'D192', 'D565', 'D130', 'D78', 'D550',
    'D153', 'D148', 'D105', 'D144', 'D173', 'D540', 'D70', 'D11', 'D569', 'D89',
    'D61', 'D134', 'D139', 'D87', 'D67', 'D126', 'D186', 'D244', 'D170', 'D174',
    'D115', 'D110', 'D195', 'D86', 'D132', 'D27', 'D104', 'D533', 'D165', 'D508',
    'D161', 'D37', 'D576', 'D520', 'D595', 'D607', 'D82', 'D119', 'D152', 'D214',
    'D98', 'D564', 'D96', 'D568', 'D208', 'D558', 'D517', 'D574', 'D535', 'D598',
    'D118', 'D548', 'D516', 'D155', 'D73', 'D187', 'D551', 'D603', 'D59', 'D09',
    'D527', 'D84', 'D74', 'D198', 'D156', 'D113', 'D79', 'D68', 'D573', 'D50',
    'D241', 'D578', 'D519', 'D183', 'D108', 'D62', 'D590', 'D563', 'D537', 'D206',
    'D150', 'D48', 'D44', 'D91', 'D55', 'D177', 'D213', 'D585', 'D111', 'D204',
    'D580', 'D240', 'D129', 'D544', 'D54', 'D159', 'D546', 'D116', 'D562', 'D80',
    'D168', 'D29', 'D46', 'D109', 'D12', 'D16', 'D570', 'D154', 'D579', 'D171',
    'D18', 'D71', 'D60', 'D85', 'D56', 'D526', 'D203', 'D180', 'D531', 'D41',
    'D235', 'D509', 'D209', 'D163', 'D53', 'D35', 'D43', 'D609', 'D66', 'D599',
    'D547', 'D606', 'D212', 'D107', 'D140', 'D556', 'D504', 'D31', 'D600', 'D572',
    'D26', 'D141', 'D137', 'D571', 'D523', 'D106', 'D583', 'D13', 'D238', 'D242',
    'D02', 'D136', 'D76', 'D201', 'D99', 'D211', 'D146', 'D182', 'D593', 'D14',
    'D502', 'D94', 'D149', 'D243', 'D584', 'D03', 'D162', 'D157', 'D202', 'D608',
    'D101', 'D179', 'D58', 'D199', 'D88', 'D100', 'D503', 'D143', 'D69', 'D93',
    'D102', 'D524', 'D122', 'D555', 'D123', 'D191', 'D234', 'D210', 'D582', 'D83',
    'D604', 'D06', 'D160', 'D138', 'D97', 'D193', 'D77', 'D125', 'D147', 'D164',
    'D133', 'D207', 'D216', 'D514', 'D52', 'D230', 'D124', 'D506', 'D538', 'D518',
    'D158', 'D245', 'D63', 'D510', 'D178', 'D512', 'D36', 'D196', 'D507', 'D117',
    'D247', 'D95', 'D505', 'D57', 'D228', 'D189', 'D167', 'D610', 'D575', 'D597',
    'D51', 'D05', 'D200', 'D532', 'D513', 'D586', 'D539', 'D166', 'D28', 'D176',
    'D611', 'D545', 'D601', 'D65', 'D197', 'D23', 'D567', 'D17', 'D47', 'D515',
    'D596', 'D128', 'D103', 'D552', 'D581', 'D530', 'D190', 'D566', 'D522', 'D145',
    'D07', 'D588', 'D577', 'D500', 'D45', 'D135', 'D511', 'D602', 'D184', 'D121',
    'D92', 'D08', 'D81', 'D194', 'D605', 'D04', 'D32', 'D75', 'D205', 'D185', 'D10',
    'D557', 'D49', 'D529', 'D554', 'D536', 'D112', 'D589', 'D25', 'D90', 'D188',
    'D549', 'D142', 'D30', 'D21', 'D42', 'D64', 'D131', 'D169']

available_speakers = [
    "D151", "D150", "D36", "D52", "D54", "D305", "D304", "D37", "D283", "D303",
    "D307", "D51", "D245", "D302", "D309", "D56", "D248", "D79", "D301", "D284", 
    "D288", "D272", "D289", "D290", "D282", "D280", "D269", "D268", "D277", "D308",
    "D291", "D292"]

available_speakers = [
    "D245", "D240", "D242", "D241", # Females
    "D243", "D506", "D213", "D174", "D196"] #Males

#available_speakers = [
#    "D151", "D150", "D36", "D52", "D54", "D305", "D304"]


def get_sentence_subdirectories(a_dir):
    return [name for name in listdir(a_dir)
            if isdir(join(a_dir, name)) and name.startswith('S')]

class WavFileDataSource(FileDataSource):
    """Wav file data source for CMU Arctic dataset.

    The data source collects wav files from CMU Arctic.
    Users are expected to inherit the class and implement ``collect_features``
    method, which defines how features are computed given a wav file path.

    Args:
        data_root (str): Data root.
        speakers (list): List of speakers to find. Supported names of speaker
         are ``awb``, ``bdl``, ``clb``, ``jmk``, ``ksp``, ``rms`` and ``slt``.
        labelmap (dict[optional]): Dict of speaker labels. If None,
          it's assigned as incrementally (i.e., 0, 1, 2) for specified
          speakers.
        max_files (int): Total number of files to be collected.

    Attributes:
        labels (numpy.ndarray): Speaker labels paired with collected files.
          Stored in ``collect_files``. This is useful to build multi-speaker
          models.
    """

    def __init__(self, data_root, speakers, labelmap=None, max_files=None):
        for speaker in speakers:
            if speaker not in available_speakers:
                raise ValueError(
                    "Unknown speaker '{}'. It should be one of {}".format(
                        speaker, available_speakers))

        self.data_root = data_root
        self.speakers = speakers
        if labelmap is None:
            labelmap = {}
            for idx, speaker in enumerate(speakers):
                labelmap[speaker] = idx
        self.labelmap = labelmap
        self.max_files = max_files
        self.labels = None

    def collect_files(self):
        """Collect wav files for specific speakers.

        Returns:
            list: List of collected wav files.
        """
        speaker_dirs = list(
            map(lambda x: join(self.data_root, x),
                self.speakers))
        paths = []
        labels = []

        if self.max_files is None:
            max_files_per_speaker = None
        else:
            max_files_per_speaker = self.max_files // len(self.speakers)
        for (i, d) in enumerate(speaker_dirs):
            if not isdir(d):
                raise RuntimeError("{} doesn't exist.".format(d))
            for sd in get_sentence_subdirectories(d):
                files = [join(join(speaker_dirs[i], sd), f) for f in listdir(join(d, sd))]
                files = list(filter(lambda x: splitext(x)[1] == ".wav", files))
                files = sorted(files)
                files = files[:max_files_per_speaker]
                for f in files:
                    paths.append(f)
                    labels.append(self.labelmap[self.speakers[i]])

        self.labels = np.array(labels, dtype=np.int32)
        return paths

