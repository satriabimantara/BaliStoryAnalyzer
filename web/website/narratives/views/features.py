from django.template import loader
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from pandas import read_excel, concat
from random import randint
from gensim.models import FastText
from scipy.sparse import hstack
import pickle
from balinarrativenlp.packages.preprocessor.TextPreprocessor import *
from balinarrativenlp.packages.BalineseCharactersNER import BaseModel
from balinarrativenlp.packages.AliasClustering import AliasClusteringRuleBased
from balinarrativenlp.packages.CharacterClassificationModule.SentenceGrouping import SentenceGrouping
from balinarrativenlp.packages.CharacterClassificationModule.FeatureExtraction.LexiconFeatureExtraction import LexiconFeatureExtraction
from balinarrativenlp.packages.CharacterClassificationModule.FeatureExtraction.POSTagFeatureExtraction import POSTagFeatureExtraction
from balinarrativenlp.packages.CharacterClassificationModule.FeatureExtraction.WordEmbeddingFeatureExtraction import WordEmbeddingFeatureExtraction
from balinarrativenlp.packages.CharacterClassificationModule.RuleBasedLexiconClassifier import RuleBasedLexiconClassifier


# setting select options here
context = {
    'title_page': 'Features',
    'heading': 'Character Identification',
    'sub_heading': 'Choose your own model and try it!',
    'character_identification_models': [
        'SatuaNER',
        'CRF-2',
        'CRF-1',
        'SVM',
        'HMM'
    ],
    'character_classification_models': {
        'rule-based': 'Rule-based BaliSentiLex Classifier',
        'knn': 'K-Nearest Neighbor',
        'dt': 'Decision Tree',
        'svm': 'Support Vector Machine',
        'rf': 'Random Forest',
        'gnb': 'Gaussian Naive Bayes',
        'mlp': 'Multilayer Perceptron',
    },
    'chararacters_classification_features': {
        'lx-1': 'Lexicon-based',
        'lx-2': 'POS Tag',
        'lx-3': 'Lexicon-based + POS Tag',
        'wv-1': 'TF-IDF',
        'wv-2': 'Word Embedding (FastText)',
        'wv-3': 'TF-IDF + Word Embedding (FastText)'
    },
    'pairwise_distance_algorithms': {
        'ratcliff': 'Ratcliff-Obershelp algorithm',
        'jaccard': 'Jaccard similarity',
        'sorensen-dice': 'Sorensen-dice similarity',
        'jaro-distance': 'Jaro Similarity',
        'jaro-winkler': 'Jaro-Winkler similarity',
    },
    'pretrained_models': {
        'M1': 'M1: trained with the best i-th training fold',
        'M2': 'M2: trained with all the training data',
    }
}


def framework_index(request):
    context.update({
        'title_page': 'Framework',
        'heading': 'Initial Framework',
        'sub_heading': 'Input your text here!'
    })
    template = 'narratives/framework/index.html'

    return render(request, template, context)


def character_identification_view(request):
    context.update({
        'title_page': 'Features',
        'heading': 'Character Identification',
        'sub_heading': 'Choose your own model and try it!',
    })
    template = 'narratives/framework/character-identification.html'
    _reset_context_view(context, 'flag_results')
    if request.POST:
        # get story text
        story_text = request.POST.get('input_story')

        # get pretrained model
        chars_identification_model = request.POST.get(
            'input_chars_identification_model')
        pretrained_chars_identification = request.POST.get(
            'pretrained_chars_identification')

        # run character identification methods
        list_of_identified_characters = _character_identification(
            story_text=story_text,
            model_name=chars_identification_model,
            pretrained_model=pretrained_chars_identification
        )
        context.update({
            'flag_results': {
                'title': 'List of identified character entities',
                'list_of_identified_characters': list_of_identified_characters,
                'pretrained_model_name': f"{chars_identification_model} [{pretrained_chars_identification}]"
            }
        })

    return render(request, template, context)


def alias_clustering_view(request):
    context.update({
        'title_page': 'Features',
        'heading': 'Alias Clustering',
        'sub_heading': 'Choose your own model and try it!',
    })
    template = 'narratives/framework/alias-clustering.html'
    _reset_context_view(context, 'flag_results')
    if request.POST:
        # get story text
        story_text = request.POST.get('input_story')

        # get chars identification model
        chars_identification_model = {
            'model_name': request.POST.get(
                'input_chars_identification_model'),
            'pretrained_model': request.POST.get(
                'pretrained_chars_identification')
        }

        # get the pairwise distance string matching
        pairwise_distance = request.POST.get('input_alias_clustering_pairwise')

        # run alias_clustering methods
        dict_character_groups = _alias_clustering(
            story_text=story_text,
            chars_identification_model=chars_identification_model,
            pairwise_distance=pairwise_distance
        )
        context.update({
            'flag_results': {
                'title': 'List of identified character groups',
                'dict_character_groups': dict_character_groups,
                'pretrained_model_name': f"{chars_identification_model['model_name']} [{chars_identification_model['pretrained_model']}] | {context['pairwise_distance_algorithms'][pairwise_distance]}",
            }
        })

    return render(request, template, context)


def character_classification_view(request):
    context.update({
        'title_page': 'Features',
        'heading': 'Character Classification',
        'sub_heading': 'Choose your own model and try it!',
    })
    template = 'narratives/framework/character-classification.html'
    _reset_context_view(context, 'flag_results')
    if request.POST:
        # get the input from user
        story_text = request.POST.get('input_story')

        # check dulu apakah button dari home page initial framework
        if request.POST.get('btnTryInitialFramework') == 'submit':
            post_update = request.POST.copy()
            post_update.update({
                'input_chars_identification_model': 'SatuaNER',
                'pretrained_chars_identification': 'M2',
                'input_chars_classification_model': 'svm',
                'input_chars_classification_features': 'wv-1',
                'input_alias_clustering_pairwise': 'sorensen-dice',
            })
            request.POST = post_update

        chars_identification_model = {
            'model_name': request.POST.get(
                'input_chars_identification_model'),
            'pretrained_model': request.POST.get(
                'pretrained_chars_identification')
        }
        chars_classification_model = {
            'model_name': request.POST.get(
                'input_chars_classification_model'),
            'features': request.POST.get(
                'input_chars_classification_features')
        }
        pairwise_distance = request.POST.get('input_alias_clustering_pairwise')

        # Run the process
        classified_character_groups = _character_classification(
            request,
            story_text=story_text,
            chars_identification_model=chars_identification_model,
            pairwise_distance=pairwise_distance,
            chars_classification_model=chars_classification_model
        )
        context.update({
            'flag_results': {
                'title': 'List of identified character groups with characterization',
                'classified_character_groups': classified_character_groups,
                'pretrained_model_name': f"{chars_identification_model['model_name']} [{chars_identification_model['pretrained_model']}] | {context['pairwise_distance_algorithms'][pairwise_distance]} | {context['character_classification_models'][chars_classification_model['model_name']]} [{context['chararacters_classification_features'][chars_classification_model['features']]}]"
            }
        })

    return render(request, template, context)


def _preprocessing_text_input(text_input):
    preprocessed_text_input = TextPreprocessor.convert_special_characters(
        text_input)
    preprocessed_text_input = TextPreprocessor.normalize_words(
        preprocessed_text_input)
    preprocessed_text_input = TextPreprocessor.remove_tab_characters(
        preprocessed_text_input)
    preprocessed_text_input = TextPreprocessor.remove_special_punctuation(
        preprocessed_text_input)
    preprocessed_text_input = TextPreprocessor.remove_punctuation(
        preprocessed_text_input)
    preprocessed_text_input = TextPreprocessor.remove_exclamation_words(
        preprocessed_text_input)
    preprocessed_text_input = TextPreprocessor.add_enter_after_period_punctuation(
        preprocessed_text_input)
    preprocessed_text_input = TextPreprocessor.remove_whitespace_LT(
        preprocessed_text_input)
    preprocessed_text_input = TextPreprocessor.remove_whitespace_multiple(
        preprocessed_text_input)
    return preprocessed_text_input


def _character_identification(
    story_text,
    model_name,
    pretrained_model
):
    """
    By default we will use M2 pretrained model for character identification. M2 model is retrieved from model that is trained using whole train dataset
    """
    # 1. preprocess text input
    preprocessed_text_input = _preprocessing_text_input(story_text)

    # 2. call the function to identify character and pass your preprocessed text and your pretrained model
    trained_model = pickle.load(open(
        settings.BALINLP_PRETRAINED_CHARS_IDENTIFICATION_MODELS+f"/{model_name}/pretrained_best_model.pkl", 'rb'))
    if pretrained_model == 'M2':
        trained_model = trained_model['optimal_best_all_train_model']
    else:
        trained_model = trained_model['optimal_best_fold_model']['model']

    list_of_identified_characters = BaseModel.identify_characters(
        preprocessed_story=preprocessed_text_input,
        pretrained_model=trained_model
    )
    return list_of_identified_characters


def _alias_clustering(story_text, chars_identification_model, pairwise_distance):
    list_of_identified_characters = _character_identification(
        story_text=story_text,
        model_name=chars_identification_model['model_name'],
        pretrained_model=chars_identification_model['pretrained_model']
    )

    # run our AliasClustering algorithm
    aliasclustering_model = AliasClusteringRuleBased(
        pairwise_distance=pairwise_distance,
    )
    aliasclustering_model.fit(list_of_identified_characters)
    dict_character_groups = aliasclustering_model.cluster()
    dict_character_groups = dict([
        (f"Character Group-{idx+1}", v)
        for idx, (k, v) in enumerate(dict_character_groups.items())
    ])
    return dict_character_groups


def _character_classification(
    request,
    story_text,
    chars_identification_model,
    pairwise_distance,
    chars_classification_model
):
    """
    Function untuk klasifikasi penokohan dari daftar kelompok tokoh yang terdeteksi dari teks cerita yang dimasukkan.
    <Default>
    - Sementara hanya pretrained model M1 yang bisa digunakan.
    - Tidak menggunakan coreference resolution pada saat sentence grouping.

    <Input>
    - story_text: Teks cerita berbahasa Bali.
        -> Type: <Str>
    - chars_identification_model: Model identifikasi tokoh yang digunakan.
        -> Type: <Dict>
            ->(model_name, pretrained_model)
    - pairwise_distance: Metode pairwise distance string similarity yang digunakan untuk proses alias clustering.
        -> Type: <Str>
    -> chars_classification_model: Model klasifikasi tokoh yang digunakan. 
        -> Type: <Dict>
            ->(model_name, text_features)

    """
    # Run Processes
    # 1. preprocessing text story and convert into list of sentence
    preprocessed_text_input = _preprocessing_text_input(story_text)
    sentences = [
        sentence.strip() for sentence in preprocessed_text_input.split('\\n')
    ]
    sentences.pop()

    # 2. run identifikasi tokoh alias clustering
    dict_list_character_groups = _alias_clustering(
        story_text=story_text,
        chars_identification_model=chars_identification_model,
        pairwise_distance=pairwise_distance
    )
    # 3. run sentence grouping
    context_sentence_extractor = SentenceGrouping(
        add_characterization_column=False
    )
    df_chars_with_context_sentence = context_sentence_extractor.fit(
        sentences=sentences,
        characters_alias_clustering=dict_list_character_groups
    ).predict()

    # 4. preprocessing context sentence in df
    df_chars_with_context_sentence['preprocessed_context_sentence'] = df_chars_with_context_sentence['GroupedSentences'].apply(
        TextPreprocessor.case_folding)
    df_chars_with_context_sentence['preprocessed_context_sentence'] = df_chars_with_context_sentence['GroupedSentences'].apply(
        TextPreprocessor.remove_stop_words)
    df_chars_with_context_sentence['GroupedSentences'].apply(
        TextPreprocessor.remove_exclamation_words)
    df_chars_with_context_sentence['GroupedSentences'].apply(
        TextPreprocessor.normalize_words)
    df_chars_with_context_sentence['preprocessed_context_sentence'] = df_chars_with_context_sentence['preprocessed_context_sentence'].apply(
        TextPreprocessor.remove_period_punctuation)
    df_chars_with_context_sentence['preprocessed_context_sentence'] = df_chars_with_context_sentence['preprocessed_context_sentence'].apply(
        lambda x: x.replace('-', ' '))
    df_chars_with_context_sentence['preprocessed_context_sentence'] = df_chars_with_context_sentence['preprocessed_context_sentence'].apply(
        TextPreprocessor.remove_whitespace_LT)
    # df_chars_with_context_sentence['preprocessed_context_sentence'].apply(
    #     TextPreprocessor.lemmatize_text)

    # 5. Feature Extraction sesuai masukan user
    X_features = None
    filename_pretrained_model = None
    feature_code, feature_number = chars_classification_model['features'].split(
        '-')
    if feature_code == 'lx':
        # load balisentilex
        baliSentiLex = read_excel(
            settings.BALINLP_BALISENTILEX+"/cleaned_baliSentiLex_english_merge.xlsx")
        baliSentiLex_terms = list(baliSentiLex['balinese_term'])
        baliSentiLex_tags = list(baliSentiLex['tags'])
        baliSentiLex_positive_scores = list(
            baliSentiLex['AveragePositiveScore'])
        baliSentiLex_negative_scores = list(
            baliSentiLex['AverageNegativeScore'])
        lexicon_extractor = LexiconFeatureExtraction(
            lexicons={
                'lexicon_terms': baliSentiLex_terms,
                'lexicon_tags': baliSentiLex_tags,
                'lexicon_positive_scores': baliSentiLex_positive_scores,
                'lecixon_negative_scores': baliSentiLex_negative_scores
            },
            text_column_to_extracted='preprocessed_context_sentence'
        )
        postag_extractor = POSTagFeatureExtraction(
            text_column_to_extracted='preprocessed_context_sentence')
        if int(feature_number) == 1:
            X_features = lexicon_extractor.fit(
                df_chars_with_context_sentence).transform()
            filename_pretrained_model = '/lexicon/m1/TF1_TF2_pretrained_baseline_m1.pkl'
        elif int(feature_number) == 2:
            X_features = postag_extractor.fit(
                df_chars_with_context_sentence).transform()
            filename_pretrained_model = '/lexicon/m1/TF3_pretrained_baseline_m1.pkl'
        elif int(feature_number) == 3:
            X_lexicon = lexicon_extractor.fit(
                df_chars_with_context_sentence).transform()
            X_postag = postag_extractor.fit(
                df_chars_with_context_sentence).transform()
            X_features = concat([X_lexicon, X_postag], axis=1)
            filename_pretrained_model = '/lexicon/m1/TF1_TF2_TF3_pretrained_baseline_m1.pkl'
    elif feature_code == 'wv':
        if int(feature_number) == 1:
            filename_pretrained_model = '/wordvector/m1/TF4_pretrained_baseline_m1.pkl'
        else:
            # load pretrained word embedding
            fasttext_model = FastText.load(
                os.path.join(
                    settings.BALINLP_PRETRAINED_WORDEMBEDDING_MODELS, '300_fasttext_model.bin')
            )
            wordembedding_extractor = WordEmbeddingFeatureExtraction(
                pretrained_word_embedding=fasttext_model
            )
            X_wordembedding = wordembedding_extractor.fit(
                df_chars_with_context_sentence).transform()
            if int(feature_number) == 2:
                filename_pretrained_model = '/wordvector/m1/TF5_pretrained_baseline_m1.pkl'
            else:
                filename_pretrained_model = '/wordvector/m1/TF4_TF5_pretrained_baseline_m1.pkl'

    # 5. Using pretrained character classification model
    model_name = chars_classification_model['model_name']
    full_model_name = context['character_classification_models'][model_name]
    trained_chars_classification = pickle.load(
        open(
            settings.BALINLP_PRETRAINED_CHARS_CLASSIFICATION_MODELS +
            filename_pretrained_model,
            'rb'
        )
    )
    y_train_encoder = pickle.load(
        open(
            f"{settings.BALINLP_PRETRAINED_LABEL_ENCODER}/without_coreference_splitted_characterization.pkl",
            'rb'
        )
    )['train']['y_encoder']
    if feature_code == 'lx':
        if model_name == 'rule-based':
            if int(feature_number) == 2:
                _reset_context_view(context, 'flag_results')
                messages.error(
                    request, 'Please select Lexicon-based Features for Rule-based BaliSentiLex Classifier')
                return redirect('narratives:chars_classify')
            rulebased_model = RuleBasedLexiconClassifier(
                label_encoder_model=y_train_encoder
            )
            y_pred = y_train_encoder.inverse_transform(
                rulebased_model.fit(X_features).predict()
            )
        else:
            trained_chars_classification = trained_chars_classification[
                full_model_name]['pretrained_model']
            pretrained_model = trained_chars_classification['estimator']
            feature_scaler = trained_chars_classification['feature_scaler']
            X_features = feature_scaler.transform(X_features.to_numpy())
            y_pred = y_train_encoder.inverse_transform(
                pretrained_model.predict(X_features)
            )

    elif feature_code == 'wv':
        if model_name == 'rule-based':
            _reset_context_view(context, 'flag_results')
            messages.error(
                request, 'Please select Lexicon-based Features for Rule-based BaliSentiLex Classifier')
            return redirect('narratives:chars_classify')
        else:
            trained_chars_classification = trained_chars_classification[
                full_model_name]['pretrained_model']
            pretrained_model = trained_chars_classification['estimator']

            if int(feature_number) == 1:
                # ekstraksi TF-IDF feature
                feature_extractor = trained_chars_classification['feature_extractor']['TF4']
                X_features = feature_extractor.transform(
                    df_chars_with_context_sentence['preprocessed_context_sentence']).toarray()
            elif int(feature_number) == 2:
                # ekstraksi word embedding feature
                X_features = X_wordembedding.copy()
            elif int(feature_number) == 3:
                # ekstraksi TF-IDF + word embedding
                feature_extractor = trained_chars_classification['feature_extractor']['TF4']
                X_tfidf = feature_extractor.transform(
                    df_chars_with_context_sentence['preprocessed_context_sentence'])
                X_features = hstack([
                    X_tfidf, X_wordembedding
                ]).toarray()
            y_pred = y_train_encoder.inverse_transform(
                pretrained_model.predict(X_features)
            )

    # 6. Create character groups and characterization pairs
    classified_character_groups = list()
    def r(): return randint(0, 255)
    for idx, (key_character_group, alias_characters) in enumerate(dict_list_character_groups.items()):
        classified_character_groups.append({
            'CharacterGroupsKey': key_character_group,
            'CharacterAliases': "; ".join(alias_characters),
            'Characterization': y_pred[idx],
            'Color': '#%02X%02X%02X' % (r(), r(), r())
        })

    return classified_character_groups


def _reset_context_view(context, deleted_keys):
    if type(deleted_keys) is list:
        for deleted_key in deleted_keys:
            context.pop(deleted_key, None)
    elif type(deleted_keys) is str:
        context.pop(deleted_keys, None)
