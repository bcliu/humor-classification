import inquirer
from subprocess import call
import sys
import click
import csv

def erase_lines(num_lines):
    for i in range(num_lines):
        sys.stdout.write("\033[F")  # back to previous line
        sys.stdout.write("\033[K")  # clear line

def load_existing_annotations(path):
    labels = {}
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            labels[int(row[0])] = int(row[1])
    return labels

def load_sentences_or_categories(path, file_has_header=False):
    """Loads sentences or categories into a dictionary of idx-sentence mappings

    :param path: Path to the file to be loaded
    :param file_has_header: Whether the file contains a header row that should be ignored
    :return: Dictionary that maps index of sentence/category to sentence/category
    """
    sentences = {}
    with open(path) as csv_file:
        if file_has_header:
            csv_file.readline()
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sentences[int(row[0])] = row[1]
    return sentences

def save_annotations_to_file(existing_annotations, new_annotations, save_to):
    print(f'You annotated {len(new_annotations)} sentences this time! Saving...')
    existing_annotations.update(new_annotations)
    all_annotations = list(existing_annotations.items())
    all_annotations.sort(key=lambda item: item[0])

    with open(save_to, 'w') as file:
        file.seek(0)
        for annotation in all_annotations:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(list(annotation))

    print('Saved')

def print_stats_by_category(category_id_to_name, existing_annotations, new_annotations):
    counts = []
    for category_id in category_id_to_name:
        count = sum(existing_annotations[x] == category_id for x in existing_annotations.keys()) +\
            sum(new_annotations[x] == category_id for x in new_annotations.keys())
        counts.append(f'{category_id_to_name[category_id]}: {count}')
    print('\nCurrent statistics: ' + ', '.join(counts))

@click.command()
@click.option('--categories-def', help='Path to categories definition CSV file', required=True)
@click.option('--input-path', help='Path to input sentences', required=True)
@click.option('--output-path', help='Path to output labels file. Can be nonempty.', required=True)
@click.option('--verbose', is_flag=True, default=False)
@click.option('--start-pos', help='Start labeling position', type=int)
@click.option('--end-pos', help='End labeling position', type=int)
def annotator(categories_def, input_path, output_path, verbose, start_pos, end_pos):
    call('clear')

    idx2categories = load_sentences_or_categories(categories_def)
    categories2idx = {idx2categories[idx]: idx for idx in idx2categories}
    existing_annotations = load_existing_annotations(output_path)
    sentences = load_sentences_or_categories(input_path, file_has_header=True)
    choices = [name for name in categories2idx]

    new_annotations = {}

    sentences_list = list(sentences.items())
    num_sentences = len(sentences_list)
    for dict_idx in range(start_pos if start_pos is not None else 0,
                          min(end_pos, num_sentences) if end_pos is not None else num_sentences):
        sentence_idx, _ = sentences_list[dict_idx]
        sentence = sentences[sentence_idx]
        if sentence_idx in existing_annotations:
            if verbose:
                print(f'{sentence} has been annotated with "{existing_annotations[sentence_idx]}"')
            continue

        if len(new_annotations) % 10 == 0:
            print_stats_by_category(idx2categories, existing_annotations, new_annotations)

        questions = [inquirer.List('sentence_label', message=sentence, choices=choices)]
        answer = inquirer.prompt(questions)
        # Handle keyboard interrupt
        if answer is None:
            save_annotations_to_file(existing_annotations, new_annotations, output_path)
            sys.exit(0)

        category_idx = categories2idx[answer['sentence_label']]
        new_annotations[sentence_idx] = category_idx

        erase_lines(len(choices) + 1)

    save_annotations_to_file(existing_annotations, new_annotations, output_path)

if __name__ == '__main__':
    annotator()
