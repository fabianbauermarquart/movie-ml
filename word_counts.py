from pyspark import SparkContext

import argparse


def get_word_counts_from_file(in_file: str, out_path: str) -> None:
    """
    Get a list of word_counts from a given file using PySpark, and write the output to the provided output path.

    :param in_file: Text file to read in.
    :param out_path: Output path to write the resulting word counts to.
    """
    # New Spark context
    spark_context = SparkContext('local', 'PySpark Word Count')

    # Read data from text file:
    # - consider only lines starting with 'BG:'
    # - apply whitespace tokenization
    words = spark_context \
        .textFile(in_file) \
        .filter(lambda line: line.startswith('BG:')) \
        .flatMap(lambda line: line.split(" "))

    # count each word's occurrence
    word_counts = words \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda word: word)

    # output
    word_counts.saveAsTextFile(out_path)


if __name__ == '__main__':
    # Create an argument parse to read command line arguments
    parser = argparse.ArgumentParser(description='Get a list of word_counts from a given file using PySpark, '
                                                 'and write the output to the provided output path.')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input text filename')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output path name')

    args = parser.parse_args()

    # Call main function with provided arguments
    get_word_counts_from_file(in_file=args.input,
                              out_path=args.output)
