import tgt
import os
import glob
from pathlib import Path

def read_word_alignment(read_textgrid):
    read_tier_words = read_textgrid.get_tier_by_name('words')
    n_words = len(read_tier_words)
    words = []
    start_time = []
    end_time = []
    for i_w in range(n_words):
        words.append(read_tier_words[i_w].text)
        start_time.append(read_tier_words[i_w].start_time)
        end_time.append(read_tier_words[i_w].end_time)

    return words, start_time, end_time


def get_textgrid_sa(mfa_file, merge_shorter=0.15, pause_tokens=[""]):
    read_textgrid = tgt.read_textgrid(mfa_file, include_empty_intervals=False)
    [words, start_time, end_time] = read_word_alignment(read_textgrid)
    assert len(words) == len(start_time) == len(end_time) #
    stack = []
    out_words = []
    for i in range(len(words)):

        if words[i] in pause_tokens:  # pause skip
            continue

        if stack:
            if start_time[i] - stack[-1][-1] > merge_shorter:
                # determine how much long is the pause

                out_words.append([words[i]])
                stack.append([start_time[i], end_time[i]])
            else: #i번째 start time에서 i-1번째 end time을 뺐을 때 간격이 merge shorter보다 작은 경우 i-1이랑 통합시킴
                stack[-1][-1] = end_time[i] #end time을 바꿈
                out_words[-1].append(words[i]) #out_words의 마지막 words(i-1번째 word)에 i번째 word를 더해줌

        else:

            stack.append([start_time[i], end_time[i]]) #stack, out_words 초기화 부분
            out_words.append([words[i]])

    # tmp = ""
    # for i in range(len(out_words)):
    #     if out_words[i] == out_words[-1]:
    #         tmp += out_words[i]
    #     else:
    #         tmp += out_words[i] + ' '

    return stack, out_words


def build_hashtable_textgrid(textgrid_dir):
    hashtab = {}

    mfa_files = glob.glob(os.path.join(textgrid_dir, "**/*.TextGrid"), recursive=True)
    #textgrid 폴더는 코퍼스와 같은 형식으로 만들어짐.
    for f in mfa_files:
        filename = Path(f).stem
        if filename not in hashtab.keys():
            hashtab[filename] = f
        else:
            raise EnvironmentError # all files are unique

    return hashtab


# testing ##

if __name__ == "__main__":
    dict_align = build_hashtable_textgrid("/home/sam/Downloads/librispeech_alignments/")