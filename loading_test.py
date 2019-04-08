from sktime.experiments.data import DataLoader


dl = DataLoader(dts_dir='data/datasets', task_types='TSC')

while True:
    try:
        value = dl.load(load_test=True)
    except StopIteration:
        print('no more')
        break
    print(value)

