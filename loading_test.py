from sktime.experiments.data import DataLoader


dl = DataLoader(dts_dir='data/datasets')

while True:
    try:
        value = dl.load_ts(task_type='TSC', load_test=True)
    except StopIteration:
        print('no more')
        break
    print(value)

