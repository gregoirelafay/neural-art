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
