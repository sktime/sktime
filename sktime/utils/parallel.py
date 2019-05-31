from joblib import Parallel

def generate_dataset_batches(X, parallel):
    num_instances = X.shape[0]
    n_jobs = parallel.n_jobs
    if n_jobs < 0:
        n_jobs = parallel._effective_n_jobs()
    start = 0
    batch_size = num_instances / n_jobs
    end = start + batch_size - 1
    start = int(start)
    end = int(end)
    for i in range(0, n_jobs - 1):
        yield start, end
        start += batch_size
        end += batch_size
        start = int(start)
        end = int(end)
    end = num_instances - 1
    start = int(start)
    end = int(end)
    yield start, end
