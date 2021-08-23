import os
from shutil import copyfile
import pandas as pd

def get_cs_class(csv_path, target):
    return pd.read_csv(os.path.join(csv_path,target + "_class.txt"), header=None, delim_whitespace=True)

def get_cs_train_val(csv_path, target, class_=None):

    if not isinstance(class_, pd.DataFrame):
        class_ = get_cs_class(csv_path, target)

    cs_train = pd.read_csv(os.path.join(csv_path, target + "_train.csv"), header=None)
    cs_train["split"] = "train"

    cs_val = pd.read_csv(os.path.join(csv_path, target + "_val.csv"), header=None)
    cs_val["split"] = "val"

    cs = pd.concat([cs_train, cs_val], ignore_index=True)
    cs.columns = ["path", target + "_id", "cs-split-" + target]

    cs[target] = cs[target + "_id"].apply(lambda x: class_.loc[x][1])
    cs["style-from-path"] = cs["path"].apply(lambda x: x.split('/')[0])
    cs["artist-from-path"] = cs["path"].apply(lambda x: x.split('/')[1].split('_')[0])
    cs["title-from-path"] = cs["path"].apply(lambda x: x.split('/')[1].split('_')[1])

    return cs

def get_data(csv_path, image_path, rm_duplicate=True):

    cs_style = get_cs_train_val(csv_path, 'style')
    cs_genre = get_cs_train_val(csv_path, 'genre')
    cs_artist = get_cs_train_val(csv_path, 'artist')

    # There is one duplicata inside genre_train.csv / genre_test.csv
    # One image labelled with two genres
    if rm_duplicate:
        if not cs_genre[cs_genre["path"].duplicated(keep='first')].empty:
            cs_genre.drop(cs_genre[cs_genre["path"].duplicated(keep='first')].index,
                          inplace=True)

    movement_list = [i for i in os.listdir(image_path)
                     if i != '.DS_Store']

    movement = []
    image = []
    artist = []
    title = []
    path = []

    for g in movement_list:
        files = os.listdir(os.path.join(image_path,g))
        movement.extend([g]*len(files))
        image.extend(files)
        artist.extend(list(map(lambda x: x.split('_')[0], files)))
        title.extend(list(map(lambda x: x.split('_')[1], files)))
        path.extend(list(map(lambda x: g + '/' + x, files)))

    data = pd.DataFrame({"path": path, "movement": movement, "artist": artist, "title": title, "image": image})

    data = data.merge(cs_genre[["path", "genre", "cs-split-genre"]], on="path",  how="outer")
    data = data.merge(cs_style[["path", "cs-split-style"]], on="path",  how="outer")
    data = data.merge(cs_artist[["path", "cs-split-artist"]], on="path",  how="outer")

    return data

def get_sample(data, input_path, output_path, target='movement',
               n=1000, random_state=123, replace=False, create_directory=False):
    if target == 'genre':
        data.dropna(axis=0, subset=[target], inplace=True)

    sample = data.groupby(by=target).sample(n=n, random_state=random_state, replace=replace)

    if create_directory:
        for i, j in sample.iterrows():
            old_path = os.path.join(input_path, j.path)
            new_path = os.path.join(output_path,
                                    f"{os.path.basename(output_path)}-{target}-{n}",
                                    eval(f"j.{target}"), j.title)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            copyfile(old_path,new_path)

        sample["path"] = sample[[target,"image"]].apply(lambda x: "/".join(x),axis=1)

    return sample
