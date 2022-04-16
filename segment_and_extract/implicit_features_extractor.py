

def adding_fingertips_statistics(palm_df):
    tips_ros = [ 12, 10, 8, 6]
    return {'fingertips' : palm_df.iloc[tips_ros]['mean']}

def adding_inbetween_finger_statistics(palm_df):
    inbetween_ros = [ 11, 9, 7]
    return { 'inbetween': palm_df.iloc[inbetween_ros]['mean'] }

def adding_center_palm_statistics(palm_df):
    return { 'all_palm_center': palm_df.iloc[0:4]['mean'] }

def manage_add_implicit_data(palm_df):
    implicit_data = {}
    for func in func_implicit:
        implicit_data.update(func(palm_df))
    return implicit_data

func_implicit = [
    adding_fingertips_statistics, adding_inbetween_finger_statistics, adding_center_palm_statistics
]
