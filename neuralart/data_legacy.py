import os
from shutil import copyfile
import pandas as pd
import splitfolders


def get_train_val_test_directory(output_dir, input_dir=None,
                                 csv_path=None,
                                 save_csv=False,
                                 train_ratio=0.8,
                                 val_ratio=0.1,
                                 test_ratio=0.1,
                                 seed=1337):

    if input_dir:
        splitfolders.ratio(input_dir, output_dir,
                        seed=seed,
                        ratio=(train_ratio, val_ratio, test_ratio),
                        group_prefix=None)


    if csv_path:
        split_list = [i for i in os.listdir(output_dir) if i != '.DS_Store']
        assert set(split_list) == {'train', 'val', 'test'}

        data = pd.read_csv(csv_path)
        data["split"] = None

        for split in split_list:
            movement_list = [i for i in os.listdir(os.path.join(output_dir,split)) if i != '.DS_Store']
            for movement in movement_list:
                image_list = [i for i in os.listdir(os.path.join(output_dir,split,movement)) if i != '.DS_Store']
                for image in image_list:
                    data.loc[data["file_name"]==image,"split"] = split

        if save_csv:
            data.to_csv(os.path.splitext(csv_path)[0] + '_split.csv', index=False)

        return data


def get_dataset(data, target="movement", class_=None, n=None, strategy='drop',
                random_state=123, output_path=None, keep_genre=False, input_image=None, output_image=None):
    '''
    Returns a dataframe after sampling the classes and merging / dropping the images

        Parameters:
            data : DataFrame
                A DataFrame coming from get_data()
            target : String
                The target of the dataset: it may be 'movement', 'genre' or 'artist'
            class_ : Dict | None
                Merge or drop classes according to dict: {'class_name': 'new_class_name'}
                will merge the class 'class_name' into the class 'new_class_name';
                {'class_name': None} will drop the class 'class_name'
            n : Int | None
                Number of samples per classes. If a class has a number of images n_class < n,
                the sampling is done according to the 'Strategy' argument. If 'n' is 'None', no
                sampling is done.
            strategy : string
                The strategy to follow to sample a class having a number of images n_class < n.
                'Drop': Drop classes with n_class < n
                'Replace': Allow sampling of the same image more than once.
                'Max': Take all available images of the class
            output_path : string | None
                Output path of the csv file. If 'output_path' is 'None', no csv is created

        Returns:
            data : pd.DataFrame
    '''
    data_tmp = data.copy()

    if target == 'genre' or keep_genre:
        data_tmp.dropna(axis=0, subset=['genre'], inplace=True)
        keep_genre = True


    if class_:
        class2drop = [key for key, val in class_['merging'].items() if not val]
        class2keep = {key: val for key,
                      val in class_['merging'].items() if val}
        data_tmp = data_tmp[data_tmp[target].apply(lambda x: x  not in class2drop)]
        data_tmp[target] = data_tmp[target].apply(lambda x: class2keep.get(x, x))

    if n:
        if strategy=='replace':
            data_tmp = data_tmp.groupby(by=target).sample(n=n,
                                              random_state=random_state,
                                              replace=True)
        if strategy=='drop':
            class2keep = (data_tmp.groupby(by=target)[target].count() >= n).to_dict()
            data_tmp = data_tmp[data_tmp[target].apply(lambda x: class2keep.get(x,False))]
            data_tmp = data_tmp.groupby(by=target).sample(n=n,
                                              random_state=random_state,
                                              replace=False)

        if strategy=='max':
            class2sample  = (data_tmp.groupby(by=target)[target].count() >= n).to_dict()
            data2sample = data_tmp[data_tmp[target].apply(lambda x: class2sample.get(x,False))]
            data2keep = data_tmp[data_tmp[target].apply(lambda x: not class2sample.get(x,False))]

            data2sample = data2sample.groupby(by=target).sample(n=n, random_state=random_state,
                                                                replace=False)

            data_tmp = pd.concat([data2keep,data2sample])

    if output_path:
        csv_name = f"{os.path.basename(output_path)}-{target}-genre_{keep_genre}-class_{data_tmp[target].nunique()}"
        if class_: csv_name = f"{csv_name}-{class_['name']}"
        if n: csv_name = f"{csv_name}-n_{n}_{strategy}"
        save_csv(data_tmp[["file_name","movement","genre","artist"]], output_path,f"{csv_name}.csv")

    return data_tmp


def get_data(csv_path,
             image_path,
             rm_csv_duplicate=True,
             rm_image_duplicate=True,
             output_path=None
             ):
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
            output_path : String | None
                Output path of the csv file containing the output of the get_data() function.
                If 'output_path' is 'None', no csv is created

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
    if rm_csv_duplicate:
        if not cs_genre[cs_genre["path"].duplicated(keep='first')].empty:
            cs_genre.drop(
                cs_genre[cs_genre["path"].duplicated(keep='first')].index,
                inplace=True)

    movement_list = [i for i in os.listdir(image_path) if i != '.DS_Store']

    movement = []
    file_name = []
    artist = []
    title = []
    path = []

    for g in movement_list:
        files = os.listdir(os.path.join(image_path, g))
        movement.extend([g] * len(files))
        file_name.extend(files)
        artist.extend([x.split('_')[0] for x in files])
        title.extend([x.split('_')[1] for x in files])
        path.extend([g + '/' + x for x in files])
        #artist.extend(list(map(lambda x: x.split('_')[0], files)))
        #title.extend(list(map(lambda x: x.split('_')[1], files)))
        #path.extend(list(map(lambda x: g + '/' + x, files)))

    data = pd.DataFrame({
        "path": path,
        "movement": movement,
        "artist": artist,
        "title": title,
        "cs_file_name": file_name
    })

    data['movement'] = data['movement'].str.lower()

    data['file_name'] = (data['movement'].str.replace(
        "_", "-") + '_' + data['cs_file_name']).str.lower()

    data = data.merge(cs_genre[["path", "genre", "cs_split_genre"]],
                      on="path",
                      how="outer")
    data = data.merge(cs_style[["path", "cs_split_style"]],
                      on="path",
                      how="outer")
    data = data.merge(cs_artist[["path", "cs_split_artist"]],
                      on="path",
                      how="outer")

    data.rename(columns={"path": "cs_path"},inplace=True)

    if rm_image_duplicate:
        data.drop(data[data['cs_file_name'].duplicated(keep=False)].index,inplace=True)

    if output_path:
        file_name = f"{os.path.basename(output_path)}-movement-class_{data['movement'].nunique()}"
        save_csv(data, output_path,f"{file_name}.csv")

    return data


def create_dataset_directory(data, chan_image_path, dataset_dir_path, dataset_dir_name, flat=True):
    j = 0
    if flat:
        for i in data.iterrows():
            os.makedirs(os.path.join(dataset_dir_path,
                        dataset_dir_name), exist_ok=True)
            copyfile(os.path.join(chan_image_path, i[1]['cs_path']), os.path.join(
                dataset_dir_path, dataset_dir_name, i[1]['file_name']))
            j +=1

            if not j % 2500:
                print(f"{j} images copied")

    else:
        for i in data.iterrows():
            os.makedirs(os.path.join(dataset_dir_path, dataset_dir_name,
                                     i[1]['movement']),
                        exist_ok=True)
            copyfile(os.path.join(chan_image_path, i[1]['cs_path']), os.path.join(
                dataset_dir_path, dataset_dir_name, i[1]['movement'], i[1]['file_name']))
            j +=1

            if not j % 2500:
                print(f"{j} images copied")

    files = os.listdir(os.path.join(dataset_dir_path, dataset_dir_name))

    print(f"Done: {j} image(s) copied, {len(files)} image(s) in the folder")

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
    cs.columns = ["path", target + "_id", "cs_split_" + target]

    cs[target] = cs[target + "_id"].apply(lambda x: class_.loc[x][1])
    cs["style_from_path"] = cs["path"].apply(lambda x: x.split('/')[0])
    cs["artist_from_path"] = cs["path"].apply(
        lambda x: x.split('/')[1].split('_')[0])
    cs["title_from_path"] = cs["path"].apply(
        lambda x: x.split('/')[1].split('_')[1])

    return cs


def save_csv(data, csv_path, filename):
    data.to_csv(os.path.join(csv_path, filename), index=False)
