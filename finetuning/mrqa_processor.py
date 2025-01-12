""" Adapted from HuggingFace code for SQuAD """

from tqdm import tqdm
from mosaic_augment_utils import mosaic_npairs_single_qac_aug, context_shuffle_aug, concat_lorem_ipsum, concat_coherent_text
import os
import json


class MRQAExample:
    """
        A single training/test example for the MRQA dataset, as loaded from disk.

        Args:
            qas_id: The example's unique identifier
            question_text: The question string
            question_tokens: The tokenized question
            context_text: The context string
            context_tokens: The tokenized context
            answer_text: The answer string
            answer_tokens: The tokenized answer
            start_position_character: The character position of the start of the answer
            answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        """

    def __init__(
            self,
            qas_id,
            question_text,
            question_tokens,
            context_text,
            context_tokens,
            answer_text,
            start_position_character,
            answers=[],
            is_impossible=False
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.question_tokens = question_tokens
        self.context_text = context_text
        self.context_tokens = context_tokens
        self.answer_text = answer_text
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        self.is_impossible = is_impossible
        doc_tokens = []
        char_to_word_offset = []
        for i, (token, token_char_position) in enumerate(context_tokens):
            doc_tokens.append(token)
            char_to_word_offset.extend([i] * len(token))

            # Verifying this is not the last token:
            if i >= len(context_tokens) - 1:
                continue
            next_token_start_position = context_tokens[i + 1][1]
            chars_to_next_token = next_token_start_position - len(char_to_word_offset)
            assert chars_to_next_token >= 0
            if chars_to_next_token > 0:
                char_to_word_offset.extend([i] * chars_to_next_token)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None:
            try:
                self.start_position = char_to_word_offset[start_position_character]
                self.end_position = char_to_word_offset[
                    min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
                ]
            except:
                print(start_position_character)
                print(char_to_word_offset)
                import pdb; pdb.set_trace()



class MRQAProcessor:
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"

    def df_to_MRQA_list(seld, df):

        examples = []

        for i, row in tqdm(df.iterrows()):
            context = row["context"]
            context_tokens = row["context_tokens"]
            for qa in row["qas"]:
                qas_id = qa["id" if "id" in qa else "qid"]
                question_text = qa["question"]
                question_tokens = qa["question_tokens"]
                try:
                    answer = qa["detected_answers"][0]
                except:
                    print(i)
                    continue

                answer_text = " ".join(
                    [c_t[0] for c_t in context_tokens[answer['token_spans'][0][0]: answer['token_spans'][0][1] + 1]])
                start_position_character = answer["char_spans"][0][0]
                answers = []

                examples.append(MRQAExample(qas_id=qas_id, question_text=question_text, question_tokens=question_tokens,
                                            context_text=context, context_tokens=context_tokens,
                                            answer_text=answer_text,
                                            start_position_character=start_position_character, answers=answers))
        return examples

    #####################################################################################
    ##################################### ERAN-AUG ######################################
    #####################################################################################
    def augment_input_data(self, input_data, aug_type, single_qac=False, max_seq_length=328):
        if aug_type.startswith('mosaic'):
            mosaic_kind, pairs, final_single_qac_triplets = aug_type.split('-')
            if mosaic_kind == "mosaic":
                aug_df = mosaic_npairs_single_qac_aug(input_data, pairs=pairs, final_single_qac_triplets=single_qac, crop=False)
            if mosaic_kind == "mosaiccrop":
                # Write crop mosaicing here
                aug_df = mosaic_npairs_single_qac_aug(input_data, pairs=pairs, final_single_qac_triplets=single_qac, crop=True)

        elif aug_type.startswith('context-shuffle'):
            aug_df = context_shuffle_aug(input_data)

        elif aug_type.startswith('lorem-ipsum'):
            aug_df = concat_lorem_ipsum(input_data, max_seq_length)

        elif aug_type.startswith('lorem-ipsum-double'):
            aug_df = concat_lorem_ipsum(input_data, both=True, max_seq_length=max_seq_length)

        elif aug_type.startswith('concat-coherent-text'):
            aug_df = concat_coherent_text(input_data, max_seq_length)

        else:
            import pdb; pdb.set_trace()

        # Verify order is different each time it's called
        print(aug_df['context'][:4])

        examples = self.df_to_MRQA_list(aug_df)

        return examples


    def create_examples(self, input_data, set_type, aug, single_qac=False, max_seq_length=328):
        """
        :param input_data: list of the jsonl MRQA formatted examples
        :param set_type: string "train" or else
        :param aug: augmentation type to use
        :return: list of MRQA Examples
        """
        is_training = set_type == "train"
        examples = []

        if aug and is_training:
            if aug != 'baseline': # Don't do any augmentation if passed baseline
                return self.augment_input_data(input_data, aug, single_qac, max_seq_length)

        for entry in tqdm(input_data):
            context = entry["context"]
            context_tokens = entry["context_tokens"]
            for qa in entry["qas"]:
                qas_id = qa["id" if "id" in qa else "qid"]
                question_text = qa["question"]
                question_tokens = qa["question_tokens"]

                if is_training:
                    answer = qa["detected_answers"][0]
                    answer_text = " ".join([c_t[0] for c_t in context_tokens[answer['token_spans'][0][0]: answer['token_spans'][0][1] + 1]])
                    start_position_character = answer["char_spans"][0][0]
                    answers = []
                else:
                    start_position_character = None
                    answer_text = None
                    answers = [{"text": " ".join([c_t[0] for c_t in context_tokens[answer['token_spans'][0][0]: answer['token_spans'][0][1] + 1]])}
                               for answer in qa["detected_answers"]]

                examples.append(MRQAExample(qas_id=qas_id, question_text=question_text, question_tokens=question_tokens,
                                            context_text=context, context_tokens=context_tokens,
                                            answer_text=answer_text,
                                            start_position_character=start_position_character, answers=answers))
        return examples

    def get_train_examples(self, data_dir, filename=None, aug='', single_qac=False, max_seq_length=328):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            print(reader.readline())
            input_data = [json.loads(line) for line in reader]
        return self.create_examples(input_data, "train", aug, single_qac, max_seq_length)

    def get_dev_examples(self, data_dir, filename=None, max_seq_length=328):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            print(reader.readline())
            input_data = [json.loads(line) for line in reader]

        # No aug neede for test set
        aug = ''
        return self.create_examples(input_data, "dev", aug, max_seq_length)
