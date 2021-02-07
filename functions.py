import pandas as pd
import numpy as np

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
        print(f"not enough iterations. try more iterations/ Current num_iters = {num_iters}")
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
#         print('using local sampling')
#         print('in generate random sample')
#         print(f"d = {len(d)}")
#         print(f"train data type = {type(train_data)}")
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



#old version, very slow
def hypothesises_to_feat_matrix(pos_hyps: pd.DataFrame, neg_hyps: pd.DataFrame, trainX: pd.DataFrame):
    pos_features = [f'pos_feat_{feat_num}' for feat_num in pos_hyps.index]
#     neg_features = [f'neg_feat_{feat_num}' for feat_num in neg_hyps[pi]s.index]
    
    result = pd.DataFrame(index=trainX.index, columns=pos_features)# + neg_features)
    
    for i in range(trainX.shape[0]):
        for pi in range(pos_hyps.shape[0]):

            feat_repr = utils.similarity(obj, pos_hyps.iloc[pi])
            is_included = pos_hyps.iloc[pi].equals(feat_repr)
            if is_included:
                result.loc[i, f'pos_feat_{pi}'] = 1
            else:
                result.loc[i, f'pos_feat_{pi}'] = 0
                
        for pi in range(neg_hyps.shape[0]):
            feat_repr = utils.similarity(obj, neg_hyps.iloc[pi])
            is_included = neg_hyps.iloc[pi].equals(feat_repr)
            if is_included:
                result.loc[i, f'neg_feat_{pi}'] = 1
            else:
                result.loc[i, f'neg_feat_{pi}'] = 0
    
    
    return result
    


def to_binary_repr(num: int, pos_hyps: pd.DataFrame, neg_hyps: pd.DataFrame, indices: list, trainX: pd.DataFrame):
    ind = indices[num]
    obj = trainX.loc[ind]
    pos_features = [f'pos_feat_{feat_num}' for feat_num in pos_hyps.index]
    neg_features = [f'neg_feat_{feat_num}' for feat_num in neg_hyps.index]
    features = pos_features + neg_features
    start_values = np.zeros(shape=(1, len(features)))
#     print(start_values.shape)
#     print(obj.name)
    result = pd.DataFrame(data=start_values,index=[obj.name], columns=features, dtype='int')
    ind = obj.name
    for pi in range(pos_hyps.shape[0]):
        feat_repr = utils.similarity(obj, pos_hyps.iloc[pi])
        is_included = pos_hyps.iloc[pi].equals(feat_repr)
        if is_included:
            result.loc[ind, f'pos_feat_{pi}'] = 1
        else:
            result.loc[ind, f'pos_feat_{pi}'] = 0
        
    for pi in range(neg_hyps.shape[0]):
        feat_repr = utils.similarity(obj, neg_hyps.iloc[pi])
        is_included = neg_hyps.iloc[pi].equals(feat_repr)
        if is_included:
            result.loc[ind, f'neg_feat_{pi}'] = 1
        else:
            result.loc[ind, f'neg_feat_{pi}'] = 0
                          
    return result

def transform_to_feature_matrix(pos_hyps: pd.DataFrame, neg_hyps: pd.DataFrame, trainX: pd.DataFrame, n_jobs: int = 4):
    
    indices = trainX.index
    transform_func = partial(to_binary_repr, pos_hyps=pos_hyps, 
                             neg_hyps=neg_hyps, indices=indices, trainX=trainX)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        obj_features = executor.map(transform_func, range(len(indices)))

    features = pd.concat(obj_features)#pd.DataFrame(obj_features)
    
    return features
















