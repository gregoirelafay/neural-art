# tests/test_data.py

from neuralart.data import *
import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), "..", "raw_data", "wikiart")

class TestCreateDataset:
    def test_create_dataset_class_drop(self):
        merge_test= {
            'name': 'merge_test',
            'merging': {
                'abstract_expressionism': 'abstract_expressionism',
                'action_painting': 'action_painting',
                'analytical_cubism': 'analytical_cubism',
                'art_nouveau_modern': None,
                'baroque': None,
                'color_field_painting': None,
                'contemporary_realism': None,
                'cubism': None,
                'early_renaissance': None,
                'expressionism': None,
                'fauvism': None,
                'high_renaissance': None,
                'impressionism': None,
                'mannerism_late_renaissance': None,
                'minimalism': None,
                'naive_art_primitivism': None,
                'new_realism': None,
                'northern_renaissance': None,
                'pointillism': None,
                'pop_art': None,
                'post_impressionism': None,
                'realism': None,
                'rococo': None,
                'romanticism': None,
                'symbolism': None,
                'synthetic_cubism': None,
                'ukiyo_e': None
            }
        }

        dataset_3c = create_dataset(os.path.join(
            csv_path, "wikiart-target_style-class_27.csv"),
                                 merge=merge_test,
                                 random_state=123)

        merge_test["merging"]['art_nouveau_modern'] = 'art_nouveau_modern'
        merge_test["merging"]['baroque'] = 'baroque'

        dataset_5c = create_dataset(os.path.join(
            csv_path, "wikiart-target_style-class_27.csv"),
                                    merge=merge_test,
                                    random_state=123)

        test_nbr_3c = dataset_3c["style"].nunique() == 3
        test_nbr_5c = dataset_5c["style"].nunique() == 5

        test_name_3c = set(dataset_3c["style"].unique()) == {'abstract_expressionism', 'action_painting', 'analytical_cubism'}
        test_name_5c = set(dataset_5c["style"].unique()) == {'abstract_expressionism', 'action_painting', 'analytical_cubism','art_nouveau_modern','baroque'}

        print(
            f"\n{dataset_3c['style'].nunique()} and {dataset_5c['style'].nunique()} classes in dataset, 3 and 5 expected"
        )
        print(
            f"{dataset_3c['style'].unique()} in 3 classes dataset, 'abstract_expressionism', 'action_painting' and 'analytical_cubism' expected"
        )
        print(
            f"{dataset_5c['style'].unique()} in 5 classes dataset, 'abstract_expressionism', 'action_painting', 'analytical_cubism','art_nouveau_modern','baroque'expected"
        )

        assert(all([test_nbr_3c, test_nbr_5c, test_name_3c, test_name_5c]))


    def test_create_dataset_class_merge(self):
        merge_test = {
            'name': 'merge_test',
            'merging': {
                'baroque': "baroque",
                'early_renaissance': "renaissance",
                'high_renaissance': "renaissance",
                'mannerism_late_renaissance': "renaissance",
                'northern_renaissance': "renaissance",
                'rococo': 'rococo'
            }
        }

        dataset = create_dataset(os.path.join(
            csv_path, "wikiart-target_style-class_27.csv"),
                                 merge=merge_test,
                                 random_state=123)

        print(f"\n{dataset['style'].nunique()} classes in dataset, 23 expected")
        assert (dataset["style"].nunique() == 24)
