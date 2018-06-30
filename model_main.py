import pickle
import os
import re
import json
import numpy as np
import datetime

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, LabelBinarizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from lightfm import LightFM
from collections import defaultdict

SEED = 10


def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def process_mpd(playlists_path, target_playlists, output_file, prev_songs_window):
    max_prev_song = 0
    previous_tracks = defaultdict(lambda: defaultdict(int))
    playlists_tracks = []
    playlists = []
    playlists_extra = {'name': []}
    filenames = os.listdir(playlists_path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((playlists_path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice['playlists']:
                nname = normalize_name(playlist['name'])
                playlists_extra['name'].append(nname)
                tracks = defaultdict(int)

                sorted_tracks = sorted(playlist['tracks'], key=lambda k: k['pos'])
                prev_track = []
                for track in sorted_tracks:
                    tracks[track['track_uri']] += 1
                    curr_prev_tracks = len(prev_track)
                    for i, song_in_window in enumerate(prev_track):
                        previous_tracks[song_in_window][track['track_uri']] += (i+1)/curr_prev_tracks
                        previous_tracks[track['track_uri']][song_in_window] += (i+1)/curr_prev_tracks
                        #previous_tracks[song_in_window][track['track_uri']] += 1
                        #previous_tracks[track['track_uri']][song_in_window] += 1
                        max_prev_song = max(max_prev_song, previous_tracks[track['track_uri']][song_in_window])
                        max_prev_song = max(max_prev_song, previous_tracks[song_in_window][track['track_uri']])
                    if len(prev_track) == prev_songs_window:
                        prev_track.pop(0)
                    prev_track.append(track['track_uri'])
                playlists_tracks.append(tracks)
                playlists.append(str(playlist['pid']))

    top_pop = []
    for i in previous_tracks.keys():
        top_pop.append((i, np.sum(list(previous_tracks[i].values()))))
    top_pop = sorted(top_pop, key=lambda x:x[1], reverse=True)[:10000]
    top_pop = [t[0] for t in top_pop]

    # Add playlists on testing set
    test_playlists = []
    test_playlists_tracks = []
    target = json.load(open(target_playlists))
    train_playlists_count = len(playlists)
    test_playlists_recommended_sum = []
    for playlist in target["playlists"]:
        nname = ""
        if 'name' in playlist:
            nname = normalize_name(playlist['name'])
        playlists_extra['name'].append(nname)
        playlists.append(str(playlist['pid']))
        test_playlists.append(str(playlist['pid']))
        if len(playlist['tracks']) == 0:
            test_playlists_recommended_sum.append(top_pop)
            test_playlists_tracks.append({})
            continue

        tracks = defaultdict(int)
        for track in playlist['tracks']:
            tracks[track['track_uri']] += 1

        #playlists_tracks.append(tracks)
        test_playlists_tracks.append(tracks)
        recommended_pop = defaultdict(list)
        for t in tracks.keys():
            for pt in previous_tracks[t].keys():
                if pt not in tracks:
                    recommended_pop[pt].append(previous_tracks[t][pt] /max_prev_song)
        recommended_pop_sum = [(t, np.sum(recommended_pop[t])) for t in recommended_pop.keys()]
        recommended_pop_sum = sorted(recommended_pop_sum, key=lambda x:x[1], reverse=True)
        recommended_pop_sum = [t[0] for t in recommended_pop_sum]
        test_playlists_recommended_sum.append(recommended_pop_sum)

    print ("Data loaded. Creating features matrix")

    dv = DictVectorizer()
    interaction_matrix = dv.fit_transform(playlists_tracks+[{}]*10000)

    lb = LabelBinarizer(sparse_output=True)
    pfeat_train = lb.fit_transform(playlists_extra['name'][:1000000]+[""]*10000)
    pfeat_test = lb.transform(playlists_extra['name'])

    print ("pfeat_train", pfeat_train.shape)
    print ("pfeat_test", pfeat_test.shape)

    playlist_features = pfeat_train
    # Need to hstack playlist_features
    eye = sparse.eye(playlist_features.shape[0], playlist_features.shape[0]).tocsr()
    playlist_features_concat = sparse.hstack((eye, playlist_features))

    print ("Features matrix created. Training model")
    model = LightFM(loss='warp', no_components=200, max_sampled=30, item_alpha=1e-06, user_alpha=1e-06, random_state=SEED)
    model = model.fit(interaction_matrix, user_features=playlist_features_concat, epochs=150, num_threads=32)

    # freeze the gradient and optimize held-out users
    model.item_embedding_gradients = np.finfo(np.float32).max * np.ones_like(model.item_embedding_gradients)
    model.item_bias_gradients = np.finfo(np.float32).max * np.ones_like(model.item_bias_gradients)
    model.item_alpha = 0.0
    model.user_alpha = 0.0
    model.user_embedding_gradients[:1000000,:] = np.finfo(np.float32).max * np.ones_like(model.user_embedding_gradients[:1000000,:])
    model.user_bias_gradients[:1000000] = np.finfo(np.float32).max * np.ones_like(model.user_bias_gradients[:1000000])

    # Use the trained model to get a representation of the playlists on challenge set
    interaction_matrix = dv.transform(playlists_tracks+test_playlists_tracks)
    playlist_features = pfeat_test
    playlist_features_concat = sparse.hstack((eye, playlist_features))
    model.user_embeddings[-10000:] = ((model.random_state.rand(10000, model.no_components) - 0.5) / model.no_components).astype(np.float32)
    model = model.fit_partial(interaction_matrix, user_features=playlist_features_concat, epochs=150, num_threads=32)
    print ("Model Trained")

    user_biases, user_embeddings = model.get_user_representations(playlist_features_concat)
    item_biases, item_embeddings = model.get_item_representations()

    fuse_perc = 0.7
    with open(output_file, 'w') as fout:
        print('team_info,cocoplaya,main,andres.ferraro@upf.edu', file=fout)
        for i, playlist in enumerate(test_playlists):
            playlist_pos = train_playlists_count+i
            y_pred = user_embeddings[playlist_pos].dot(item_embeddings.T) + item_biases
            topn = np.argsort(-y_pred)[:len(test_playlists_tracks[i])+4000]
            rets = [(dv.feature_names_[t], float(y_pred[t])) for t in topn]
            songids = [s for s, _ in rets if s not in test_playlists_tracks[i]]
            songids_dict = {s:1 for s in songids}
            max_score = max(len(songids), len(test_playlists_recommended_sum[i]))
            pop_sum = {s:(max_score - p) for p,s in enumerate(test_playlists_recommended_sum[i])}
            fuse_sum = []
            for p, s in enumerate(songids):
                pop_val_sum = 0 
                if s in pop_sum:
                    pop_val_sum = pop_sum[s]
                fuse_sum.append((s,((max_score - p)*fuse_perc + pop_val_sum*(1-fuse_perc) ) / 2))
            for s in pop_sum.keys():
                if s not in songids_dict:
                    fuse_sum.append((s,(pop_sum[s]*(1-fuse_perc) ) / 2))
            fuse_sum = sorted(fuse_sum, key=lambda x:x[1], reverse=True)
            print(' , '.join([playlist] + [x[0] for x in fuse_sum[:500]]), file=fout)


if __name__ == '__main__':
    playlists_file = './mpd/data/'
    target_playlists = 'eval_data/challenge_set.json'
    output_file = 'output/output_main_final_sum_window_10.csv'

    process_mpd(playlists_file, target_playlists, output_file, 10)
