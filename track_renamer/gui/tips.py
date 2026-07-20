"""Shared tooltip copy."""

TIPS = {
    "preset": "Load a saved rule preset.",
    "save_preset": "Save the current rule stack as a preset.",
    "open_folder": "Scan a folder for audio and MIDI files to rename.",
    "help": "How rules work and supported file types.",
    "delete_preset": "Delete the selected template.",
    "recursive": "Also scan files inside subfolders of the selected directory.",
    "apply_preview": (
        "Update the preview list with the current rules. "
        "Does not run Auto-detect — use Analyze for that."
    ),
    "clear_rules": "Remove all rules and reset the list.",
    "add_rule": "Choose a renaming rule to add to the stack.",
    "condition_field": "Which part of the filename to test.",
    "condition_op": "How to compare the filename against your value.",
    "condition_value": "Text or pattern to match. Supports tokens like {group} or {bpm}.",
    "rule_enable": "Enable or disable this specific modification.",
    "remove_rule": "Remove this rule from the stack.",
    "remove_category_row": "Remove this category mapping row.",
    "add_category_row": "Add a new category mapping row.",
    "prefix_field": "Text prepended when keywords in this row match.",
    "keywords_field": "Comma-separated keywords that trigger this prefix (plurals match automatically).",
    "category_color": "Click to change this category’s color.",
    "instrument_source": (
        "How the category PREFIX is chosen for each file. "
        "Hover Filename / Auto-detect / Combo for details."
    ),
    "instrument_source_filename": (
        "Filename: match keywords in the current name against the category "
        "rows below (e.g. kick → DRUMS -). No audio listening — preview "
        "updates immediately with Apply."
    ),
    "instrument_source_auto": (
        "Auto-detect: classify the audio with ML (PaSST OpenMIC-2018, "
        "20 instruments) and map the top label to a category PREFIX. "
        "Filename keywords are ignored."
    ),
    "instrument_source_combo": (
        "Combo: try Filename keywords first; if none match, fall back to "
        "Auto-detect on that file’s audio."
    ),
    "add_child_rule": "Add another operation inside this condition group.",
    "select_all": "Select every file in the preview for renaming.",
    "deselect_all": "Deselect every file in the preview.",
    "only_changed": "Show only files that will be modified by the current rules.",
    "file_checkbox": "Include or exclude this file when applying renames.",
    "cancel": "Close without applying any changes.",
    "rename": "Permanently rename the selected files on disk.",
    "remove_text": "Text to remove from the filename.",
}
