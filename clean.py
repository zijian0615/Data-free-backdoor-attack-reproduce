import pandas as pd
#

def data_clean(contact_info_file, other_info_file):
    df1 = pd.read_csv(contact_info_file)
    df2 = pd.read_csv(other_info_file)
    result = pd.merge(df1, df2, left_on='respondent_id', right_on='id')
    result = result.drop(columns='id')
    result.dropna(inplace=True)
    result = result[~result['job'].str.contains('insurance|Insurance')]
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input1', help='the path to 1st input file (CSV)')
    parser.add_argument('input2', help='the path to 2nd input file (CSV)')
    parser.add_argument('output', help='the path to output file (CSV)')

    args = parser.parse_args()

    cleaned = data_clean(args.input1, args.input2)
    print("Shape of the output file:", cleaned.shape)
    cleaned.to_csv(args.output, index=False)