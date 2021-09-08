import os
from shutil import copyfile
import pandas as pd
from sklearn.model_selection import train_test_split


def create_dataset(csv_file_name, target="style", merge=None, n=None, strategy='drop', keep_genre=True, flat=False,
                   val_ratio=None, test_ratio=None, random_state=123, csv_output_path=None, chan_image_folder_path=None, image_folder_output_path=None):
    '''
    Create a new dataset from Chan's Wikiart dataset after merging, sampling and dropping some classes/images

        Parameters:
            csv_file_name: String
                A path to a csv file containing all the information about Chan's Wikiart Dataset.
                Use the function get_chan_data() to generate this csv.
            target: String
                The main target of the dataset: it may be 'style', 'genre' or 'artist'.
                The train test split is performed according to the specified target.
            merge: Dict | None
                A dictionary specifying the classes to merge/drop.
                {'class_name_1': 'class_name_2'}: merge the class 'class_name_1' into the class 'new_class_name_2'.
                {'class_name_3': None}: drop the class 'class_name_3'.
                If 'merge' is 'None', no merge/drop is done
            n: Int | None
                Sample the classes by randomly selecting n images per classes.
                If a class has a number of images smaller than n, the sampling is done according to the 'Strategy' argument.
                If 'n' is 'None', no sampling is done.
            strategy: string
                The strategy used to sample a class having a number of images smaller tan n.
                'Drop': Drop the classes
                'Replace': Allow sampling of the same image more than once.
                'Max': Take all available images of the class
            keep_genre: Bool
                If True, keep only the images having a "genre" label.
                If False, keep all the images.
            flat: Bool
                If False, create the new directory containing all the images of the new dataset according to the following formats:
                    - If train, val, test split is performed:
                        folder/
                            train/
                                class1/
                                    img1.jpg
                                    img2.jpg
                                    ...
                                class2/
                                    ...
                            val/
                                ...
                            test/
                                ...
                    - If no split is performed:
                        folder/
                            class1/
                                img1.jpg
                                img2.jpg
                                ...
                            class2/
                                ...
                If True, create the new directory according to the following format:
                    folder/
                        img1.jpg
                        img2.jpg
                        ...
            chan_image_folder_path: string
                Path to the folder containing images from Chan's wikiart dataset
            image_folder_output_path
                Path to create the new directory containing all the images of the new dataset
                If 'image_folder_output_path' is 'None', no directory is created
            csv_output_path : string | None
                Path to save a new csv file containing all the information about the new dataset
                If 'csv_output_path' is 'None', no csv is created

        Returns:
            df: pd.DataFrame
            A dataframe containing all the information of the new dataset
    '''
    df = pd.read_csv(csv_file_name)

    assert (val_ratio and test_ratio) or (
        not val_ratio and not test_ratio), "Please set both val_ratio and test_ratio"

    if target == 'genre' or keep_genre:
        df.dropna(axis=0, subset=['genre'], inplace=True)
        keep_genre = True

    # Merging
    if merge:
        class2drop = [key for key, val in merge['merging'].items() if not val]
        class2keep = {key: val for key,
                      val in merge['merging'].items() if val}
        df = df[df[target].apply(lambda x: x  not in class2drop)]
        df[target] = df[target].apply(lambda x: class2keep.get(x, x))

    # Sampling
    if n:
        if strategy=='replace':
            df = df.groupby(by=target).sample(n=n,
                                              random_state=random_state,
                                              replace=True)
        if strategy=='drop':
            class2keep = (df.groupby(by=target)[target].count() >= n).to_dict()
            df = df[df[target].apply(lambda x: class2keep.get(x,False))]
            df = df.groupby(by=target).sample(n=n,
                                              random_state=random_state,
                                              replace=False)

        if strategy=='max':
            class2sample  = (df.groupby(by=target)[target].count() >= n).to_dict()
            data2sample = df[df[target].apply(lambda x: class2sample.get(x,False))]
            data2keep = df[df[target].apply(lambda x: not class2sample.get(x,False))]

            data2sample = data2sample.groupby(by=target).sample(n=n, random_state=random_state,
                                                                replace=False)

            df = pd.concat([data2keep,data2sample])

    # Train, val, test split
    if val_ratio and test_ratio:
        df_train, df_val_test = train_test_split(
            df,
            test_size=val_ratio + test_ratio,
            random_state=random_state,
            stratify=df[target])
        df_val, df_test = train_test_split(df_val_test,
                                           test_size=test_ratio /
                                           (val_ratio + test_ratio),
                                           random_state=random_state,
                                           stratify=df_val_test[target])

        df_train.insert(1, "split", ["train"]*df_train.shape[0])
        df_val.insert(1, "split", ["val"]*df_val.shape[0])
        df_test.insert(1, "split", ["test"]*df_test.shape[0])

        df = pd.concat([df_train, df_val, df_test],ignore_index=True)

    if not flat:
        if val_ratio and test_ratio:
            df.insert(0, "image_path", [os.path.join(i[1]["split"], i[1][target], i[1]["image_name"])
                                for i in df[[target, "split", "image_name"]].iterrows()])
        else:
            df.insert(0, "image_path", [os.path.join(i[1][target], i[1]["image_name"])
                                for i in df[[target, "image_name"]].iterrows()])

    # Save csv
    if csv_output_path:
        csv_name = f"wikiart-target_{target}-class_{df[target].nunique()}-keepgenre_{keep_genre}"
        if merge: csv_name = f"{csv_name}-merge_{merge['name']}"
        if n: csv_name = f"{csv_name}-n_{n}_{strategy}"
        csv_name = f"{csv_name}-flat_{flat}"
        save_csv(df, csv_output_path, f"{csv_name}.csv")

    # Create new directory
    if chan_image_folder_path and image_folder_output_path:
        dir_name = f"wikiart-target_{target}-class_{df[target].nunique()}-keepgenre_{keep_genre}"
        if merge: dir_name = f"{dir_name}-merge_{merge['name']}"
        if n: dir_name = f"{dir_name}-n_{n}_{strategy}"
        dir_name = f"{dir_name}-flat_{flat}"
        create_directory(df, chan_image_folder_path,
                         os.path.join(image_folder_output_path, dir_name))

    return df

def get_chan_data(chan_csv_folder_path,
             chan_image_folder_path,
             rm_csv_duplicate=True,
             rm_image_duplicate=True,
             csv_output_path=None
             ):
    '''
    Returns a complete dataframe containing all the information of all the files in
    Chan's wikiart dataset.

        Parameters:
            chan_csv_folder_path: string
                Path to the folder containing Chan's csv files
            chan_image_folder_path: string
                Path to the folder containing images from Chan's wikiart dataset
            rm_duplicate: bool
                Remove duplicatas from Chan's csv file
            rm_image_duplicate: bool
                Remove duplicatas from Chan's images
            csv_output_path : String | None
                Path to save a new csv file containing all the information about Chan's wikiart dataset
                If 'csv_output_path' is 'None', no csv is created

        Returns:
            data : pd.DataFrame
                Dataframe containing all the information of all the images in the wikiart dataset,
                as well as the genre labels and the train/val splits of cs-chan
    '''

    cs_style = get_chan_train_val_split(chan_csv_folder_path, 'style')
    cs_genre = get_chan_train_val_split(chan_csv_folder_path, 'genre')
    cs_artist = get_chan_train_val_split(chan_csv_folder_path, 'artist')

    # There is one duplicata inside genre_train.csv / genre_test.csv
    # One image labelled with two genres
    if rm_csv_duplicate:
        if not cs_genre[cs_genre["chan_image_path"].duplicated(keep='first')].empty:
            cs_genre.drop(
                cs_genre[cs_genre["chan_image_path"].duplicated(
                    keep='first')].index,
                inplace=True)

    style_list = [i for i in os.listdir(
        chan_image_folder_path) if i != '.DS_Store']

    style = []
    file_name = []
    artist = []
    title = []
    path = []

    for g in style_list:
        files = os.listdir(os.path.join(chan_image_folder_path, g))
        style.extend([g] * len(files))
        file_name.extend(files)
        artist.extend([x.split('_')[0] for x in files])
        title.extend([x.split('_')[1] for x in files])
        path.extend([g + '/' + x for x in files])

    data = pd.DataFrame({
        "chan_image_path": path,
        "style": style,
        "artist": artist,
        "title": title,
        "chan_image_name": file_name
    })

    data['style'] = data['style'].str.lower()

    data['image_name'] = (data['style'].str.replace(
        "_", "-") + '_' + data['chan_image_name']).str.lower()

    data = data.merge(cs_genre[["chan_image_path", "genre", "chan_split_genre"]],
                      on="chan_image_path",
                      how="outer")
    data = data.merge(cs_style[["chan_image_path", "chan_split_style"]],
                      on="chan_image_path",
                      how="outer")
    data = data.merge(cs_artist[["chan_image_path", "chan_split_artist"]],
                      on="chan_image_path",
                      how="outer")

    if rm_image_duplicate:
        data.drop(data[data['chan_image_name'].duplicated(keep=False)].index,inplace=True)

    data = data[["image_name", "style", "genre", "artist", "title", "chan_image_path",
                 "chan_image_name", "chan_split_style", "chan_split_genre", "chan_split_artist"]]

    if csv_output_path:
        os.makedirs(os.path.join(csv_output_path), exist_ok=True)
        file_name = f"wikiart-target_style-class_{data['style'].nunique()}.csv"
        save_csv(data, csv_output_path, file_name)

    return data

def get_chan_train_val_split(chan_csv_folder_path, target, merge=None):

    if not isinstance(merge, pd.DataFrame):
        merge = get_chan_annotations(chan_csv_folder_path, target)

    cs_train = pd.read_csv(os.path.join(chan_csv_folder_path, target + "_train.csv"),
                           header=None)
    cs_train["split"] = "train"

    cs_val = pd.read_csv(os.path.join(chan_csv_folder_path, target + "_val.csv"),
                         header=None)
    cs_val["split"] = "val"

    cs = pd.concat([cs_train, cs_val], ignore_index=True)
    cs.columns = ["chan_image_path", target + "_id", "chan_split_" + target]

    cs[target] = cs[target + "_id"].apply(lambda x: merge.loc[x][1])
    cs["style_from_path"] = cs["chan_image_path"].apply(lambda x: x.split('/')[0])
    cs["artist_from_path"] = cs["chan_image_path"].apply(
        lambda x: x.split('/')[1].split('_')[0])
    cs["title_from_path"] = cs["chan_image_path"].apply(
        lambda x: x.split('/')[1].split('_')[1])

    return cs


def get_chan_annotations(chan_csv_folder_path, target):
    return pd.read_csv(os.path.join(chan_csv_folder_path, target + "_class.txt"),
                       header=None,
                       delim_whitespace=True)


def save_csv(data, csv_output_path, file_name):
    data.to_csv(os.path.join(csv_output_path,file_name), index=False)


def create_directory(data, chan_image_folder_path, image_folder_output_path):
    j = 0
    os.makedirs(os.path.join(image_folder_output_path), exist_ok=True)

    for i in data.iterrows():
        os.makedirs(os.path.dirname(
            os.path.join(image_folder_output_path, i[1]['image_path'])),
                    exist_ok=True)

        copyfile(os.path.join(chan_image_folder_path, i[1]['chan_image_path']),
                 os.path.join(image_folder_output_path, i[1]['image_path']))
        j += 1

        if not j % 2500:
            print(f"{j} images copied")

    files = os.listdir(image_folder_output_path)

    print(f"Done: {j} image(s) copied")
