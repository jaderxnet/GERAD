import json


class NpEncoder(json.JSONEncoder):
    '''
        Class To encoder json files 
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class NpDecorder(json.JSONDecoder):
    '''
        Class To decoder json files 
    '''

    def default(self, obj):
        if isinstance(obj, int):
            return np.integer(obj)
        if isinstance(obj, float):
            return np.floating(obj)
        if isinstance(obj, list):
            return np.ndarray(obj)
        return super(NpDecorder, self)
