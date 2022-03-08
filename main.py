import streamlit as st
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


## Loading model 
model = load_model("cut_position.h5")

## Loading df

df = pd.read_pickle('df_cut_position.pkl')


def prediction (seq):

    SAMPLES_PER_GROUP = 7320
    MAX_VOCAB_SIZE = 50
    SEQUENCE_SIZE = 50
    EMBEDDING_SIZE = 50

    # tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(df['sequence'])

    sequences = tokenizer.texts_to_sequences(seq)
    x_test = pad_sequences(sequences, maxlen=SEQUENCE_SIZE, padding='post', truncating='post', value=0)

    # predict
    predict_x = model.predict(x_test) 
    result = np.argmax(predict_x,axis=1)

    return result


# App 
def main():
    st.title('Model Cut position')
    st.write('''## Paste your sequence to predict the cut postion''')

    # space to user put sequence to predict cut postion 

    seq = st.sidebar.text_input('Sequence protein', 'AUGCCC')
    st.write('The sequence is', seq)

    if st.sidebar.button("Predict"): 
        result = prediction([[seq]]) 
        st.success('Your sequence have cut position in {} AA'.format(result))
        print(result)


if __name__=='__main__': 
    main()
