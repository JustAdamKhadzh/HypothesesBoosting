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


#данную функцию переписали, чтобы соответствовала общей логике подсчета
def generate_local_area(obj: pd.Series):
    
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
        left_val, right_val = left_val - eps, right_val + eps
        local_obj[feat] = (left_val, right_val)
    local_obj = pd.Series(local_obj)
    local_obj.name = index
    return local_obj

#здесь высчитвается локальная область, которая передается функции генерации семпла
#сейчас расширение области делается в отрыве от самих данных. Нужно предусмотреть ограничения на расширение признака
#Чтобы не получилось, что возраст отрицательный. Можно передавать в качестве параметра вектор минимальных и максимальных значений
#для всех признаков
def find_opt_local_area(obj: pd.Series, train_data: pd.DataFrame,
                       frac: float = 0.15, num_iters=10):
    
    """
    generates optimal local area. Tries iteratively to generate local area
    that have at least len(train_data) * frac train objects or using at most num_iterations
    iterations.
    returns pd.Series if found opt local area or None
    """
    #нужно запихнуть сюда код, который занимается именно поиском области,
    #а в generate_local_sample оставить именно генерацию семпла по области
    iters = num_iters
    is_found = False
    d_local_area = obj
    objects_count = 0
    objects_count_thresh = int(train_data.shape[0] * frac)
    print(f"start generating local area")
    while objects_count < objects_count_thresh and iters > 0:
        print(f"itr: {iters}")
        d_local_area = generate_local_area(obj=d_local_area)
        d_local_objects = is_included_in_repr(d=d_local_area, train_data=train_data)
        if d_local_objects is not None:
            objects_count = d_local_objects.shape[0]
            print(f"d_local_objects = {objects_count}")
            if objects_count > objects_count_thresh:
                print(f"found opt local area for object")
                is_found = True
                break
        else:
            print(f'found 0 objects in this local area')
        iters -= 1 
    if not is_found:
        print(f"not enough iterations. try more iterations/ Current num_iters = {num_iters}")
    return d_local_area if is_found else None

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
        print('using local sampling')
        local_objects = is_included_in_repr(d=d, train_data=train_data)
        if local_objects is None:
            print(f'cannot generate sample. Got 0 train data in local area')
            raise NotImplementedError('Wrong local sampling area')
        inds = np.random.RandomState().choice(local_objects.index, replace=False, size=sample_size)
        sample = local_objects.loc[inds]
    else:
        print('using random sampling')
        inds = np.random.RandomState().choice(train_data.index, replace=False, size=sample_size)
        sample = train_data.loc[inds].copy()
    return sample

def check_criterion(d: pd.Series, train_data: pd.DataFrame, hypothesis_criterion: str, d_other_objects: pd.DataFrame,
                    other_data: pd.DataFrame, alpha: float):
            
    other_data_size = other_data.shape[0]
    d_other_objects_size = d_other_objects.shape[0]
    d_other_objs_thresh = int(other_data_size * alpha)
    result_hypothesis = None

    if hypothesis_criterion == 'contr_class':
        if d_other_objects_size <= d_other_objs_thresh:
            result_hypothesis = d
        else:
            pass#reject
        
    if hypothesis_criterion == 'both_classes':
    #дополнительно смотрим какие объекты target(рассматриваемого на этой итерации) класса попадают в паттерн d
    #!!!!Все таки здесь нужно добавить еще член соотношения классов
        d_target_objects = is_included_in_repr(d, train_data=train_data)
        d_target_objects_size = d_target_objects.shape[0]
        d_target_objs_thresh = int(d_target_objects_size * alpha)
        result_hypothesis = None
        if d_other_objects_size <= d_target_objs_thresh:
            result_hypothesis = d
#         result_hypothesis = d if d_other_objects_size <= d_target_objs_thresh else None

    return result_hypothesis






















