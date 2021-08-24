import os
from shutil import copyfile
import pandas as pd


def save_csv(data, csv_path, filename):
    data.to_csv(os.path.join(csv_path, filename), index=False)


def save_directory(data, target, input_path, output_path, n=None):
    directory_name = f"{os.path.basename(output_path)}-{target}-class_{data[target].nunique()}"
    if n: directory_name = f"{directory_name}-n_{n}"
    for i, j in data.iterrows():
        old_path = os.path.join(input_path, j.path)
        new_path = os.path.join(output_path, directory_name,
                                eval(f"j.{target}"), j.title)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        copyfile(old_path, new_path)

    data["path"] = data[[target, "image"]].apply(lambda x: "/".join(x), axis=1)

def get_dataset(data, target="movement", class_=None, n=None, strategy='drop',
                random_state=123, output_path=None):

    data_tmp = data.copy()

    if target == 'genre':
        data_tmp.dropna(axis=0, subset=[target], inplace=True)


    if class_:
        class2drop = [key for key, val in class_.items() if not val]
        class2keep = {key:val for key, val in class_.items() if val}
        data_tmp = data_tmp[data_tmp[target].apply(lambda x: x  not in class2drop)]
        data_tmp[target] = data_tmp[target].apply(lambda x: class2keep.get(x, x))

    if n:
        if strategy=='replace':
            data_tmp = data_tmp.groupby(by=target).sample(n=n,
                                              random_state=random_state,
                                              replace=True)
        if strategy=='drop':
            class2keep = (data_tmp.groupby(by=target)[target].count() > n).to_dict()
            data_tmp = data_tmp[data_tmp[target].apply(lambda x: class2keep.get(x,False))]
            data_tmp = data_tmp.groupby(by=target).sample(n=n,
                                              random_state=random_state,
                                              replace=False)

        if strategy=='max':
            class2sample  = (data_tmp.groupby(by=target)[target].count() > n).to_dict()
            data2sample = data_tmp[data_tmp[target].apply(lambda x: class2sample.get(x,False))]
            data2keep = data_tmp[data_tmp[target].apply(lambda x: not class2sample.get(x,False))]

            data2sample = data2sample.groupby(by=target).sample(n=n, random_state=random_state,
                                                                replace=False)

            data_tmp = pd.concat([data2keep,data2sample])

    if output_path:
        file_name = f"{os.path.basename(output_path)}-{target}-class_{data_tmp[target].nunique()}"
        if n: file_name = f"{file_name}-n_{n}"
        save_csv(data_tmp, output_path,f"{file_name}.csv")


    return data_tmp

def get_sample(data,
               input_path,
               output_path,
               target='movement',
               n=50,
               random_state=123,
               replace=False,
               create_directory=False,
               create_csv=False):

    datata_tmp = data.copy()

    if target == 'genre':
        datata_tmp.dropna(axis=0, subset=[target], inplace=True)

    sample = datata_tmp.groupby(by=target).sample(n=n,
                                                  random_state=random_state,
                                                  replace=replace)

    if create_directory:
        save_directory(sample, target, input_path, output_path, n=n)

    if create_csv:
        save_csv(
            sample, output_path,
            f"{os.path.basename(output_path)}-{target}-class_{sample[target].nunique()}-n_{n}.csv"
        )

    return sample


def get_cs_class(csv_path, target):
    return pd.read_csv(os.path.join(csv_path, target + "_class.txt"),
                       header=None,
                       delim_whitespace=True)


def get_cs_train_val(csv_path, target, class_=None):

    if not isinstance(class_, pd.DataFrame):
        class_ = get_cs_class(csv_path, target)

    cs_train = pd.read_csv(os.path.join(csv_path, target + "_train.csv"),
                           header=None)
    cs_train["split"] = "train"

    cs_val = pd.read_csv(os.path.join(csv_path, target + "_val.csv"),
                         header=None)
    cs_val["split"] = "val"

    cs = pd.concat([cs_train, cs_val], ignore_index=True)
    cs.columns = ["path", target + "_id", "cs-split-" + target]

    cs[target] = cs[target + "_id"].apply(lambda x: class_.loc[x][1])
    cs["style-from-path"] = cs["path"].apply(lambda x: x.split('/')[0])
    cs["artist-from-path"] = cs["path"].apply(
        lambda x: x.split('/')[1].split('_')[0])
    cs["title-from-path"] = cs["path"].apply(
        lambda x: x.split('/')[1].split('_')[1])

    return cs


def get_data(csv_path,
             image_path,
             rm_duplicate=True,
             merge=None,
             create_csv=False,
             create_directory=False):
    '''
    Returns a complete dataframe containing all the information of all the files in
    the wikiart dataset, as well as the genre labels and the train/val splits
    of cs-chan

        Parameters:
            csv_path : string
                Path to the csv files of cs-chan
            image_path : string
                Path to the images of the wikiart dataset
            rm_duplicate : bool
                Remove a duplicata in the csv files of cs-chan
            merge : dict or None (default)
                merge one or more movements according to the parameters of the
                "merge" dictionary: {"old_movement_name": "new_movement_name",}.
                If merge is None, no merger is carried out
            create_csv : bool
                If true, export the result of the get_data() function to a csv file
            create_directory : bool
                Create a new image directory with a new file architecture based on
                the merge argument. If merge is "None", create_directory will be set to False

        Returns:
            data : pd.DataFrame
                Dataframe containing all the information of all the images in the wikiart dataset,
                as well as the genre labels and the train/val splits of cs-chan
    '''

    cs_style = get_cs_train_val(csv_path, 'style')
    cs_genre = get_cs_train_val(csv_path, 'genre')
    cs_artist = get_cs_train_val(csv_path, 'artist')

    # There is one duplicata inside genre_train.csv / genre_test.csv
    # One image labelled with two genres
    if rm_duplicate:
        if not cs_genre[cs_genre["path"].duplicated(keep='first')].empty:
            cs_genre.drop(
                cs_genre[cs_genre["path"].duplicated(keep='first')].index,
                inplace=True)

    movement_list = [i for i in os.listdir(image_path) if i != '.DS_Store']

    movement = []
    image = []
    artist = []
    title = []
    path = []

    for g in movement_list:
        files = os.listdir(os.path.join(image_path, g))
        movement.extend([g] * len(files))
        image.extend(files)
        artist.extend(list(map(lambda x: x.split('_')[0], files)))
        title.extend(list(map(lambda x: x.split('_')[1], files)))
        path.extend(list(map(lambda x: g + '/' + x, files)))

    data = pd.DataFrame({
        "path": path,
        "movement": movement,
        "artist": artist,
        "title": title,
        "image": image
    })

    data = data.merge(cs_genre[["path", "genre", "cs-split-genre"]],
                      on="path",
                      how="outer")
    data = data.merge(cs_style[["path", "cs-split-style"]],
                      on="path",
                      how="outer")
    data = data.merge(cs_artist[["path", "cs-split-artist"]],
                      on="path",
                      how="outer")

    if merge:
        data["movement"] = data["movement"].apply(lambda x: merge.get(x, x))
        data["path"] = data[["movement", "image"]].apply(lambda x: "/".join(x),
                                                         axis=1)

    if merge and create_directory:
        print("create_directory: TO DO")

    if create_csv:
        save_csv(
            data, csv_path,
            f"{os.path.basename(csv_path)}-movement-class_{data['movement'].nunique()}.csv"
        )

    return data
