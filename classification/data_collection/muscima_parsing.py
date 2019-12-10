import os
from muscima.io import parse_cropobject_list
import itertools
import numpy
import cv2
import random

# Change this to reflect wherever your MUSCIMA++ data lives
CROPOBJECT_DIR = '../MUSCIMA-pp_v1.0/v1.0/data/cropobjects_manual' # os.path.join(os.environ['HOME'], 'data/MUSCIMA++/v0.9/data/cropobjects')

cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]

###############################
# Feature extraction -- notes #
###############################
def extract_q_and_h_notes_from_doc(cropobjects):
    """Finds all ``(notehead-full, stem)`` pairs that form
    quarter or half notes. Returns two lists of CropObject tuples:
    one for quarter notes, one of half notes.

    :returns: quarter_notes, half_notes
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    notes = []
    for c in cropobjects:
        if (c.clsname == 'notehead-full') or (c.clsname == 'notehead-empty'):
            _has_stem = False
            _has_beam_or_flag = False
            stem_obj = None
            for o in c.outlinks:
                _o_obj = _cropobj_dict[o]
                if _o_obj.clsname == 'stem':
                    _has_stem = True
                    stem_obj = _o_obj
                elif _o_obj.clsname == 'beam':
                    _has_beam_or_flag = True
                elif _o_obj.clsname.endswith('flag'):
                    _has_beam_or_flag = True
            if _has_stem and (not _has_beam_or_flag):
                # We also need to check against quarter-note chords.
                # Stems only have inlinks from noteheads, so checking
                # for multiple inlinks will do the trick.
                if len(stem_obj.inlinks) == 1:
                    notes.append((c, stem_obj))

    quarter_notes = [(n, s) for n, s in notes if n.clsname == 'notehead-full']
    half_notes = [(n, s) for n, s in notes if n.clsname == 'notehead-empty']
    return quarter_notes, half_notes

def extract_w_notes_from_doc(cropobjects):
    """Finds all ``(notehead-full)`` that forms
    whole nots. Returns

    :returns: whole_notes
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    notes = []
    for c in cropobjects:
        if (c.clsname == 'notehead-empty'):
            _has_stem = False
            for o in c.outlinks:
                _o_obj = _cropobj_dict[o]
                if _o_obj.clsname == 'stem':
                    _has_stem = True
            if _has_stem:
                notes.append([c])

    return notes

def extract_e_notes_from_doc(cropobjects):
    """Finds all ``(notehead-full, stem, beam-or-flag)`` that forms
    eighth nots. Returns

    :returns: eighth_notes
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    notes = []
    for c in cropobjects:
        if (c.clsname == 'notehead-full'):
            _has_stem = False
            _has_beam_or_flag = False
            stem_obj = None
            beam_or_flag_obj = None
            for o in c.outlinks:
                _o_obj = _cropobj_dict[o]
                if _o_obj.clsname == 'stem':
                    _has_stem = True
                    stem_obj = _o_obj
                elif _o_obj.clsname == 'beam':
                    _has_beam_or_flag = True
                    beam_or_flag_obj = _o_obj
                elif _o_obj.clsname == ('8th_flag'):
                    _has_beam_or_flag = True
                    beam_or_flag_obj = _o_obj
            if _has_stem and _has_beam_or_flag:
                notes.append((c, stem_obj, beam_or_flag_obj))

    return notes


###############################
# Feature extraction -- rests #
###############################
def extract_rests_from_doc(cropobjects, rest_name):
    """Finds all ``(rest_name-rest)`` that forms
    rests. Returns

    :returns: rests
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    rests = []
    for c in cropobjects:
        if (c.clsname == rest_name + '_rest'):
            rests.append([c])

    return rests

#####################################
# Feature extraction -- accidentals #
#####################################
def extract_accs_from_doc(cropobjects, acc_name):
    """Finds all ``(acc_name)`` that forms
    accidentals. Returns

    :returns: accidentals
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    accs = []
    for c in cropobjects:
        if (c.clsname == acc_name):
            accs.append([c])

    return accs


##################################
# Getting images of each feature #
##################################
def get_image(cropobjects, margin=1):
    """Paste the cropobjects' mask onto a shared canvas.
    There will be a given margin of background on the edges."""

    # Get the bounding box into which all the objects fit
    top = min([c.top for c in cropobjects])
    left = min([c.left for c in cropobjects])
    bottom = max([c.bottom for c in cropobjects])
    right = max([c.right for c in cropobjects])

    # Create the canvas onto which the masks will be pasted
    height = bottom - top + 2 * margin
    width = right - left + 2 * margin
    canvas = numpy.zeros((height, width), dtype='uint8')

    for c in cropobjects:
        # Get coordinates of upper left corner of the CropObject
        # relative to the canvas
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        # We have to add the mask, so as not to overwrite
        # previous nonzeros when symbol bounding boxes overlap.
        canvas[_pt:_pt+c.height, _pl:_pl+c.width] += c.mask

    canvas[canvas > 0] = 1
    return canvas * 255


#################
# parsing notes #
#################
def parse_quarter_and_half_notes():
    qns_and_hns = [extract_q_and_h_notes_from_doc(cropobjects) for cropobjects in docs]

    qns = list(itertools.chain(*[qn for qn, hn in qns_and_hns]))
    hns = list(itertools.chain(*[hn for qn, hn in qns_and_hns]))
    print(len(qns), len(hns))

    qn_images = [get_image(qn) for qn in qns]
    hn_images = [get_image(hn) for hn in hns]

    random.shuffle(qn_images)
    random.shuffle(hn_images)

    q_len = len(qn_images)
    for q in range(0, q_len):
        if (q < q_len * 0.1):
            cv2.imwrite("./val/quarter_note/%05d.png" % (q), qn_images[q])
        else:
            cv2.imwrite("./train/quarter_note/%05d.png" % (q), qn_images[q])

    h_len = len(hn_images)
    for h in range(0, len(hn_images)):
        if (h < h_len * 0.1):
            cv2.imwrite("./val/half_note/%05d.png" % (h), hn_images[h])
        else:
            cv2.imwrite("./train/half_note/%05d.png" % (h), hn_images[h])
    
    print("parsed quarter and half notes")

def parse_whole_notes():
    wns = [extract_w_notes_from_doc(cropobjects) for cropobjects in docs]
    wns = list(itertools.chain(*wns))
    print(len(wns))

    wn_images = [get_image(wn) for wn in wns]

    random.shuffle(wn_images)
    w_len = len(wn_images)
    for w in range(0, w_len):
        if (w < w_len * 0.1):
            cv2.imwrite("./val/whole_note/%05d.png" % (w), wn_images[w])
        else:
            cv2.imwrite("./train/whole_note/%05d.png" % (w), wn_images[w])

    print("parsed whole notes")

def parse_eighth_notes():
    ens = [extract_e_notes_from_doc(cropobjects) for cropobjects in docs]
    ens = list(itertools.chain(*ens))
    print(len(ens))

    en_images = [get_image(en) for en in ens]

    random.shuffle(en_images)
    e_len = len(en_images)
    for e in range(0, e_len):
        if (e < e_len * 0.1):
            cv2.imwrite("./val/eighth_note/%05d.png" % (e), en_images[e])
        else:
            cv2.imwrite("./train/eighth_note/%05d.png" % (e), en_images[e])

    print("parsed eighth notes")


#################
# parsing rests #
#################
def parse_rests():
    rest_names = ["whole", "half", "quarter", "8th"]
    for rest_name in rest_names:
        rests = [extract_rests_from_doc(cropobjects, rest_name) for cropobjects in docs]
        rests = list(itertools.chain(*rests))
        print(len(rests))

        rest_images = [get_image(rest) for rest in rests]

        random.shuffle(rest_images)
        r_len = len(rest_images)
        if rest_name == "8th":
            rest_name = "eighth"
        for r in range(0, r_len):
            if (r < r_len * 0.1):
                cv2.imwrite("./val/" + rest_name + "_rest/%05d.png" % (r), rest_images[r])
            else:
                cv2.imwrite("./train/" + rest_name + "_rest/%05d.png" % (r), rest_images[r])

        print("parsed " + rest_name + " rests")


#######################
# parsing accidentals #
#######################
def parse_accs():
    acc_names = ["sharp", "flat", "natural"]
    for acc_name in acc_names:
        accs = [extract_accs_from_doc(cropobjects, acc_name) for cropobjects in docs]
        accs = list(itertools.chain(*accs))
        print(len(accs))

        acc_images = [get_image(acc) for acc in accs]

        random.shuffle(acc_images)
        a_len = len(acc_images)
        for a in range(0, a_len):
            if (a < a_len * 0.1):
                cv2.imwrite("./val/" + acc_name + "/%05d.png" % (a), acc_images[a])
            else:
                cv2.imwrite("./train/" + acc_name + "/%05d.png" % (a), acc_images[a])

        print("parsed " + acc_name + "s")


parse_accs()