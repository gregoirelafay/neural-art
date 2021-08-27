# tests/test_data.py

from neuralart.data import *
import os
import pandas as pd

csv_path = os.path.join(os.path.dirname(__file__), "..", "raw_data", "wikiart")

# DO NOT FORGET TO RUN GET_DATA TWICE AND SAVE THE RESULTS BEFORE RUNNINNG THE TESTS

class TestGetData:
    def test_get_data_nbr_image_full(self):
        DATA_FULL = pd.read_csv(os.path.join(csv_path,'data_full.csv'))
        print(f"\n{DATA_FULL.shape[0]} images in DATA_FULL, 81446 expected")
        assert DATA_FULL.shape[0] == 81446


    def test_get_data_nbr_image_rm_duplicate(self):
        DATA = pd.read_csv(os.path.join(csv_path,'data.csv'))
        print(f"\n{DATA.shape[0]} images in DATA, 78748 expected")
        assert DATA.shape[0] == 78748

class TestGetDataset:
    def test_get_dataset_class_drop(self):
        DATA = pd.read_csv(os.path.join(csv_path,'data.csv'))
        merge_test_class = {
            'name': 'merge_test_class',
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

        dataset_3c = get_dataset(DATA,
                    target="movement",
                    class_=merge_test_class,
                    random_state=123)

        merge_test_class["merging"]['art_nouveau_modern'] = 'art_nouveau_modern'
        merge_test_class["merging"]['baroque'] = 'baroque'

        dataset_5c = get_dataset(DATA,
                target="movement",
                class_=merge_test_class,
                random_state=123)

        test_nbr_3c = dataset_3c["movement"].nunique() == 3
        test_nbr_5c = dataset_5c["movement"].nunique() == 5

        test_name_3c = set(dataset_3c["movement"].unique()) == {'abstract_expressionism', 'action_painting', 'analytical_cubism'}
        test_name_5c = set(dataset_5c["movement"].unique()) == {'abstract_expressionism', 'action_painting', 'analytical_cubism','art_nouveau_modern','baroque'}

        print(f"\n{dataset_3c['movement'].nunique()} and {dataset_5c['movement'].nunique()} classes in dataset, 3 and 5 expected")
        print(
            f"{dataset_3c['movement'].unique()} in 3 classes dataset, 'abstract_expressionism', 'action_painting' and 'analytical_cubism' expected"
        )
        print(
            f"{dataset_5c['movement'].unique()} in 5 classes dataset, 'abstract_expressionism', 'action_painting', 'analytical_cubism','art_nouveau_modern','baroque'expected"
        )

        assert(all([test_nbr_3c, test_nbr_5c, test_name_3c, test_name_5c]))


    def test_get_dataset_class_merge(self):
        DATA = pd.read_csv(os.path.join(csv_path,'data.csv'))
        merge_test_class = {
            'name': 'merge_test_class',
            'merging': {
                'baroque': "baroque",
                'early_renaissance': "renaissance",
                'high_renaissance': "renaissance",
                'mannerism_late_renaissance': "renaissance",
                'northern_renaissance': "renaissance",
                'rococo': 'rococo'
            }
        }

        dataset = get_dataset(DATA,
                target="movement",
                class_=merge_test_class,
                random_state=123)

        print(f"\n{dataset['movement'].nunique()} classes in dataset, 23 expected")
        assert (dataset["movement"].nunique()==24)
