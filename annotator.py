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
    sentences = {}
    with open(path) as csv_file:
        if file_has_header:
            csv_file.readline()
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            sentences[int(row[0])] = row[1]
    return sentences


def save_annotations_to_file(existing_annotations, new_annotations, save_to):
    existing_annotations.update(new_annotations)
    all_annotations = list(existing_annotations.items())
    all_annotations.sort(key=lambda item: item[0])

    with open(save_to, 'w') as file:
        file.seek(0)
        for annotation in all_annotations:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(list(annotation))


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
def annotator(categories_def, input_path, output_path, verbose):
    call('clear')

    idx2categories = load_sentences_or_categories(categories_def)
    categories2idx = {idx2categories[idx]: idx for idx in idx2categories}
    existing_annotations = load_existing_annotations(output_path)
    sentences = load_sentences_or_categories(input_path, file_has_header=True)
    choices = [name for name in categories2idx]

    new_annotations = {}

    for sentence_idx in sentences:
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
            print(f'You annotated {len(new_annotations)} sentences this time! Saving...')
            save_annotations_to_file(existing_annotations, new_annotations, output_path)
            print('Saved')
            sys.exit(0)

        category_idx = categories2idx[answer['sentence_label']]
        new_annotations[sentence_idx] = category_idx

        erase_lines(len(choices) + 1)


if __name__ == '__main__':
    annotator()
