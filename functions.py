import pandas as pd
import numpy as np

from functools import partial
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed

def transform_to_description(data: pd.DataFrame):
    transformed_data = pd.DataFrame(columns=data.columns)
    
    for col in data:
        transformed_data[col] = data[col].apply(lambda x: (x, x))
    
    return transformed_data


def similarity(vect1: pd.Series, vect2:pd.Series):
    
    func = lambda x, y: (min(x[0], y[0]), max(x[1], y[1]))
    vect = pd.Series(map(func, vect1, vect2), index=vect1.index)
    return vect


def get_similarity_sample_repr(sample: pd.DataFrame):
    
    """
    get sample of feature represantations from pos or neg class dataset
    returns feature represantation for sample by similarity operation
    """
    pattern = None
    for i, obj in sample.iterrows():
        if pattern is None:
            pattern = obj
        else:
            pattern = similarity(pattern, obj)
    return pattern
    
#операция нахождения объектов по признаковому представлению
def is_included_in_repr(d: pd.Series, train_data: pd.DataFrame):
    """
    returns objects from train dataset(from train pos and neg data) that is included in d representation
    returns pd.DataFrame or None
    """
    
    d_list = []
    
    for i, obj in train_data.iterrows():
        feature_repr = similarity(obj, d)
        is_included = d.equals(feature_repr)
        if is_included:
            d_list.append(obj)
    d_list = pd.DataFrame(d_list) if len(d_list) > 0 else None
    return d_list


def generate_local_area(obj: pd.Series, train_min: pd.Series, train_max: pd.Series):
    
    """
    generates local area for obj expanding each feature by base value
    returns pd.Series
    """
    eps = 0
    index = obj.name
    local_obj = {}
    for feat, val in obj.items():
        left_val, right_val = val

        if isinstance(left_val, int):
            eps = 1 if abs(left_val) // 100 == 0 else 100
        elif isinstance(left_val, float):
            eps = 0.01 if abs(left_val) // 10 == 0 else 100

        if left_val > train_min[feat]:
            left_val = left_val - eps
        if right_val < train_max[feat]:
            right_val = right_val + eps
        # left_val, right_val = left_val - eps, right_val + eps
        local_obj[feat] = (left_val, right_val)
    local_obj = pd.Series(local_obj)
    local_obj.name = index
    return local_obj

def find_opt_local_area(obj: pd.Series, train_data: pd.DataFrame,
                    trainx_min: pd.Series, trainx_max: pd.Series,
                       frac: float, expanding_iters: int):

    """
    generates optimal local area. Tries iteratively to generate local area
    that have at least len(train_data) * frac train objects or using at most num_iterations
    iterations.
    returns pd.Series if found opt local area or None
    """
#     print(f'num_iters = {num_iters}')
    iters = 0
    is_found = False
    d_local_area = obj
    objects_count = 0
    objects_count_thresh = int(train_data.shape[0] * frac)
    while objects_count < objects_count_thresh and iters <= expanding_iters:
        iters += 1 
        if iters <= 100:
            if iters % 10 == 0:
                print(f"itr: {iters}")
        else:
            if iters % 100 == 0:
                print(f"itr: {iters}")
        d_local_area = generate_local_area(obj=d_local_area, train_min=trainx_min, train_max=trainx_max)
        d_local_objects = is_included_in_repr(d=d_local_area, train_data=train_data)
        if d_local_objects is not None:
            objects_count = d_local_objects.shape[0]
            #print(f"d_local_objects = {objects_count}")
            if objects_count >= objects_count_thresh:
                #print(f"found opt local area for object")
                is_found = True
                break
#         else:
#             print(f'found 0 objects in this local area')
        
    if not is_found:
        print(f"not enough iterations. try more iterations/ Current num_iters = {expanding_iters}")
    return d_local_area if is_found else None#, iters

def generate_local_sample(d: pd.Series, train_data: pd.DataFrame, sample_size: int):
    
    local_objects = is_included_in_repr(d=d, train_data=train_data)
    
    inds = np.random.RandomState().choice(local_objects.index, replace=False, size=sample_size)
    sample = local_objects.loc[inds]
    
    return sample

def generate_random_sample(train_data: pd.DataFrame, sample_size: int, d: pd.Series = None):
    
    """
    Generate random sample based on d value
    If d is None then generate random sample from train_data
    If d is not None then generate random sample from local object's area
    returns pd.DataFrame or None
    """
    
    if d is not None:
        local_objects = is_included_in_repr(d=d, train_data=train_data)
        if local_objects is None:
            print(f'cannot generate sample. Got 0 train data in local area')
            return None
        inds = np.random.RandomState().choice(local_objects.index, replace=False, size=sample_size)
        sample = local_objects.loc[inds]
    else:
        #print('using random sampling')
        inds = np.random.RandomState().choice(train_data.index, replace=False, size=sample_size)
        sample = train_data.loc[inds]
    return sample

def check_criterion(d: pd.Series, train_data: pd.DataFrame, hypothesis_criterion: str, 
                    d_other_objects: pd.DataFrame, other_data: pd.DataFrame, alpha: float):

    """Checks whether d hypothesis satisfies hypothesis_criterion

    Returns pd.Series or None
    """

    train_data_size = train_data.shape[0]
    other_data_size = other_data.shape[0]
    
    classes_ratio = train_data_size / other_data_size

    if d_other_objects is None:
        d_other_objects_size = 0
    else:
        d_other_objects_size = d_other_objects.shape[0]
        
    d_other_objs_thresh = other_data_size * alpha 

    result_hypothesis = None

    if hypothesis_criterion == 'contr_class':
        if d_other_objects_size <= d_other_objs_thresh:
            result_hypothesis = d
        #else reject hypothesis
        
    if hypothesis_criterion == 'both_classes':
        #d_target_objects не может быть None, так как область d строится по sample из train_data. Так что хотя бы
        #эти объекты будут попадать в эту область. Могут не попадать объекты только противоположного класса.
        d_target_objects = is_included_in_repr(d, train_data=train_data)
        d_target_objects_size = d_target_objects.shape[0]
        classes_ratio_thresh = alpha * classes_ratio
        
        if d_other_objects_size > 0:        
            if d_target_objects_size / d_other_objects_size > classes_ratio_thresh:
                result_hypothesis = d
        else:
            result_hypothesis = d
            
    return result_hypothesis




def generate_hypothesis(iteration: int, obj: pd.Series, object_area: pd.Series, train_data: pd.DataFrame, 
                        other_data: pd.DataFrame, sample_size: int, hypothesis_criterion: str,
                        sample_type:str, verbose: bool, alpha: float):
        
        if iteration % 100 == 0:
            print(f'iteration: {iteration}')
        
        if sample_type == 'local' and object_area is None:
            print(f'Cannot generate sample from local area. Got None as local object area param!')
            raise NotImplementedError('Wrong params for local sampling!')
    
        if sample_type == 'random' and object_area is not None:
            print(f'got misleading params values. Got sample_type = None and local object area is not None')
            return NotImplementedError('Wrong params for random sampling!')

        sample = generate_random_sample(train_data=train_data, sample_size=sample_size, d=object_area) 
        if sample is None:
            return None
        
        sample.append(obj)
        
        d = get_similarity_sample_repr(sample)
        if verbose:
            print('got feature represantation for sample')
        
        d_other_objects = is_included_in_repr(d, train_data=other_data)
        
        if verbose and d_other_objects is not None:
            print(f'got {d_other_objects.shape[0]} d_other_objects')
            print(f'thresh for hypothesis = {int(other_data.shape[0] * alpha)}')
        
        result_hypothesis = check_criterion(
                        d=d, train_data=train_data, hypothesis_criterion=hypothesis_criterion, 
                        d_other_objects=d_other_objects, other_data=other_data, alpha=alpha)

        return result_hypothesis
        

def mining_step(test_obj: pd.Series, train_pos: pd.DataFrame, train_neg: pd.DataFrame, sample_ratio: float, 
                alpha: float, hypothesis_criterion: str, sample_type: str, trainx_min: pd.Series, 
                trainx_max:pd.Series, fraction: float = 0.25, num_iters: int = 1000, expanding_iters: int = 50, 
                mining_type: str = 'pos', verbose : bool = False, n_jobs : int = 4):
    """
    hypothesis_criterion: 'contr_class', если используем базовый критерий, 
                                когда смотрится пересечение с противоположным классом(старый критерий отбора гипотез)
                           'both_classes', когда интересует пересечение по обоим классам(новый критерий отбора гипотез)
                           
    sample_type: 'random', если берем произвольную выборку интервальных представлений
                 'local', если берем произвольную выборку из локальной области
    
    returns list of hypothesises
    """
    
    train_data = train_pos if mining_type == 'pos' else train_neg
    other_data = train_neg if mining_type == 'pos' else train_pos
    sample_size = int(train_data.shape[0] * sample_ratio)
    #print('start generating hypothesises')
    
    if sample_type == 'local':
        #print(f'start searching optimal local area')
        object_area = None
        itrs = 3
        while itrs > 0:
            object_area = find_opt_local_area(obj=test_obj, train_data=train_data,
                                                    trainx_min=trainx_min, trainx_max=trainx_max, 
                                                    frac=fraction, expanding_iters=expanding_iters)
            if object_area is None:
                expanding_iters += expanding_iters
                itrs = itrs - 1
            else:
                break
        if object_area is None:
            sample_type = 'random'
    else:
        object_area = None
    
    
    mining = partial(generate_hypothesis, obj=test_obj, object_area=object_area, train_data=train_data, 
                     other_data=other_data, sample_size=sample_size, hypothesis_criterion=hypothesis_criterion,
                     sample_type=sample_type, verbose=verbose, alpha=alpha)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        hypothesises = executor.map(mining, range(num_iters))
        
    hypothesises = [res for res in hypothesises if res is not None]

    return hypothesises



def mining_temp_layer(test_obj_index, test_sample, train_pos, train_neg, sample_ratio, alpha, 
                      hypothesis_criterion, sample_type, trainx_min, trainx_max, fraction, num_iters,
                      expanding_iters, mining_type, verbose, n_jobs):
    
    test_obj = test_sample.loc[test_obj_index]
    
    print(f'object ind = {test_obj_index}')
    
    pos_hyps = mining_step(test_obj=test_obj, train_pos=train_pos, train_neg=train_neg, sample_ratio=sample_ratio, 
                alpha=alpha, hypothesis_criterion=hypothesis_criterion, sample_type=sample_type, 
                trainx_min=trainx_min, trainx_max=trainx_max, fraction=fraction, num_iters=num_iters,
                expanding_iters=expanding_iters, mining_type='pos', verbose=verbose, n_jobs=n_jobs)
    
    pos_hyps = pd.DataFrame(pos_hyps).drop_duplicates()
    pos_hyps_shape = pos_hyps.shape[0] if len(pos_hyps) > 0 else 0
    
   # print('got pos_hyps')
    
#     total_pos_objs = 0
#     if pos_hyps_shape > 0:
#         for hyp in pos_hyps.iterrows():
#             pos_objs = is_included_in_repr(hyp, train_pos)
#             pos_objs_shape = 0 if pos_objs is None else pos_objs.shape[0]
#             total_pos_objs += pos_objs_shape
    
#     print('got pos_hyps objects')

    neg_hyps = mining_step(test_obj=test_obj, train_pos=train_pos, train_neg=train_neg, sample_ratio=sample_ratio, 
                alpha=alpha, hypothesis_criterion=hypothesis_criterion, sample_type=sample_type, 
                trainx_min=trainx_min, trainx_max=trainx_max, fraction=fraction, num_iters=num_iters,
                expanding_iters=expanding_iters, mining_type='neg', verbose=verbose, n_jobs=n_jobs)
    
    neg_hyps = pd.DataFrame(neg_hyps).drop_duplicates()
    neg_hyps_shape = neg_hyps.shape[0] if len(neg_hyps) > 0 else 0

#     print('got neg_hyps')
    
#     total_neg_objs = 0
#     if neg_hyps_shape > 0:
#         for hyp in neg_hyps.iterrows():
#             neg_objs = is_included_in_repr(hyp, train_neg)
#             neg_objs_shape = 0 if neg_objs is None else neg_objs.shape[0]
#             total_neg_objs += neg_objs_shape
            
#     print('got neg_hyps objects')
            
#     print(pos_hyps_shape, neg_hyps_shape, total_pos_objs, total_neg_objs)
    
    return (pos_hyps_shape, neg_hyps_shape)#, total_pos_objs, total_neg_objs)


def mining_objs_parallel(test_sample: pd.DataFrame, train_pos: pd.DataFrame, train_neg: pd.DataFrame, 
                              sample_ratio: float, alpha: float, hypothesis_criterion: str, 
                              sample_type: str, trainx_min: pd.Series, trainx_max:pd.Series,
                              fraction: float = 0.25, num_iters: int = 1000, expanding_iters: int = 50, 
                              mining_type: str = 'pos', verbose : bool = False, num_workers: int = 4, 
                             n_jobs : int = 2):
    
    test_sample_index = test_sample.index
    
    mining = partial(mining_temp_layer, test_sample=test_sample, train_pos=train_pos, train_neg=train_neg,
                     sample_ratio=sample_ratio, alpha=alpha, hypothesis_criterion = hypothesis_criterion,
                     sample_type=sample_type, trainx_min=trainx_min, trainx_max=trainx_max, fraction=fraction, 
                     num_iters=num_iters, expanding_iters=expanding_iters, mining_type=mining_type, 
                     verbose=verbose, n_jobs=n_jobs)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        hypo_shapes = executor.map(mining, test_sample_index)
        
    hypothesises = [{ind: {'pos': pos, 
                           'neg': neg}
                    } for ind, (pos, neg) in zip(test_sample_index, hypo_shapes)
                   ]

    return hypothesises


def to_binary_repr_old(num: int, pos_hyps: pd.DataFrame, neg_hyps: pd.DataFrame, indices: list, trainX: pd.DataFrame):
    ind = indices[num]
    
    if num % 50 == 0:
        print(f'processing obj = {num}')
    obj = trainX.loc[ind]
    pos_features = [f'pos_f_{feat_num}' for feat_num in pos_hyps.index]
    neg_features = [f'neg_f_{feat_num}' for feat_num in neg_hyps.index]
    features = pos_features + neg_features
    start_values = np.zeros(shape=(1, len(features)))

    result = pd.DataFrame(data=start_values,index=[obj.name], columns=features, dtype='int')
    ind = obj.name
    for pi in range(pos_hyps.shape[0]):
        feat_repr = utils.similarity(obj, pos_hyps.iloc[pi])
        is_included = pos_hyps.iloc[pi].equals(feat_repr)
        if is_included:
            result.loc[ind, f'pos_f_{pi}'] = 1
        else:
            result.loc[ind, f'pos_f_{pi}'] = 0
        
    for pi in range(neg_hyps.shape[0]):
        feat_repr = utils.similarity(obj, neg_hyps.iloc[pi])
        is_included = neg_hyps.iloc[pi].equals(feat_repr)
        if is_included:
            result.loc[ind, f'neg_f_{pi}'] = 1
        else:
            result.loc[ind, f'neg_f_{pi}'] = 0
                          
    return result

def to_binary_repr_new(num: int, pos_hyps: pd.DataFrame, neg_hyps: pd.DataFrame, indices: list, trainX: pd.DataFrame):
    ind = indices[num]
    
    #if num % 50 == 0:
    print(f'processing obj = {num}')
    obj = trainX.loc[ind]
    pos_features = [f'pos_f_{feat_num}' for feat_num in pos_hyps.index]
    neg_features = [f'neg_f_{feat_num}' for feat_num in neg_hyps.index]
    features = pos_features + neg_features
    start_values = np.zeros(shape=(1, len(features)))

    result = pd.DataFrame(data=start_values,index=[obj.name], columns=features, dtype='int')
    ind = obj.name
    for pi, hyp in pos_hyps.iterrows():
        feat_repr = utils.similarity(obj, hyp)
        is_included = hyp.equals(feat_repr)
        if is_included:
            result.loc[ind, f'pos_f_{pi}'] = 1
        else:
            result.loc[ind, f'pos_f_{pi}'] = 0
        
    for pi, hyp in neg_hyps.iterrows():
        feat_repr = utils.similarity(obj, hyp)
        is_included = hyp.equals(feat_repr)
        if is_included:
            result.loc[ind, f'neg_f_{pi}'] = 1
        else:
            result.loc[ind, f'neg_f_{pi}'] = 0
                          
    return result

def transform_to_feature_matrix(pos_hyps: pd.DataFrame, neg_hyps: pd.DataFrame, trainX: pd.DataFrame, n_jobs: int = 4):
    
    indices = trainX.index
    transform_func = partial(
            binarization, 
            pos_hyps=pos_hyps, 
            neg_hyps=neg_hyps,
            indices=indices,
            trainX=trainX
    )
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        obj_features = executor.map(transform_func, range(len(indices)))

    features = pd.concat(obj_features)
    
    return features

def binarization(num: int, pos_hyps: pd.DataFrame, neg_hyps: pd.DataFrame, indices: list, trainX: pd.DataFrame):
    ind = indices[num]
    
    #if num % 50 == 0:
    print(f'processing obj = {num}')
    obj = trainX.loc[ind]
    pos_features = [f'pos_f_{feat_num}' for feat_num in pos_hyps.index]
    neg_features = [f'neg_f_{feat_num}' for feat_num in neg_hyps.index]
    features = pos_features + neg_features
    
    pos_repr = pos_hyps.apply(lambda hyp: hyp.equals(utils.similarity(hyp, obj)), axis=1)
    neg_repr = neg_hyps.apply(lambda hyp: hyp.equals(utils.similarity(hyp, obj)), axis=1)
    
    start_values = pos_repr.append(neg_repr).astype('int')
#     columns=[obj.name], index=features,
    result = pd.DataFrame(data=start_values, columns=[obj.name],dtype='int').T
    result.columns=features
    
    return result



# 'total_pos_objs':total_pos_objs,
#                            'total_neg_objs':total_neg_objs, , total_pos_objs, total_neg_objs










