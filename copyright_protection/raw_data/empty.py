# Created by Chen Henry Wu
import os
import datasets

_DESCRIPTION = ''
_HOMEPAGE = ''
_LICENSE = ''
_CITATION = ''
_URL = ''


class Example(datasets.GeneratorBasedBuilder):
    """The example dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        prompts_file = 'raw_data/pool.txt'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": prompts_file, "data_dir": ''},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": prompts_file, "data_dir": ''},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": prompts_file, "data_dir": ''},
            ),
        ]

    def _generate_examples(self, filepath, data_dir):
        """Yields examples."""

        # Iterate over the prompts
        for idx in range(1000):
            yield idx, {
                "id": idx,
        }
