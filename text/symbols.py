""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]
_german_phoneme_aliases = [
    "@de_a",
    "@de_aj",
    "@de_a_long",
    "@de_b",
    "@de_c",
    "@de_c_asp",
    "@de_ch",
    "@de_d",
    "@de_e",
    "@de_e_long",
    "@de_f",
    "@de_g",
    "@de_h",
    "@de_i",
    "@de_i_long",
    "@de_k",
    "@de_k_asp",
    "@de_l",
    "@de_m",
    "@de_n",
    "@de_n_syllabic",
    "@de_o",
    "@de_o_long",
    "@de_p",
    "@de_r",
    "@de_s",
    "@de_spn",
    "@de_t",
    "@de_ts",
    "@de_t_asp",
    "@de_u",
    "@de_u_long",
    "@de_v",
    "@de_w",
    "@de_z",
    "@de_ae",
    "@de_o_open",
    "@de_schwa",
    "@de_eps",
    "@de_g_soft",
    "@de_g_hard",
    "@de_i_short",
    "@de_r_uvular",
    "@de_sch",
    "@de_sil",
    "@de_u_short",
    "@de_vocalic_r",
]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _german_phoneme_aliases
    + _silences
)
