from tqdm import tqdm
from multiprocessing import cpu_count, Pool


def parallel_process(df, func, desc='Parallel processing', n_jobs=-1):
    """parallel process helper

    Args:
        df (Pandas.DataFrame or GeoDataFrame): the columns in df must contains the parmas in the func.
        func (Function): The func need to be parallel accelerated.
        desc (str, optional): [description]. Defaults to 'Parallel processing'.
        n_jobs (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """

    n_jobs = cpu_count() if n_jobs == -1 or n_jobs > cpu_count() else n_jobs
    pool = Pool(n_jobs)
    
    pbar = tqdm(total=df.shape[0], desc=desc)
    update = lambda *args: pbar.update()

    res = []
    for id, item in df.iterrows():
        tmp = pool.apply_async(func, (item,), callback=update)
        res.append(tmp)
    pool.close()
    pool.join() 

    res = [ r.get() for r in res ]
    
    return res

