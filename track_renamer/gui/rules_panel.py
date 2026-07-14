"""Rules panel — stack of condition groups and operations."""

from __future__ import annotations

import customtkinter as ctk

from track_renamer.category_palette import (
    CATEGORY_BADGE_TEXT,
    CATEGORY_PALETTE_COLORS,
    category_badge_label,
    category_color,
)
from track_renamer.engine.defaults import RULE_CATALOG, make_category_rules
from track_renamer.engine.models import (
    CategoryRule,
    Condition,
    ConditionGroup,
    OpRule,
    Rule,
)
from track_renamer.gui.tips import TIPS
from track_renamer.gui.tooltip import bind_tooltip


OP_LABELS = {
    "stripLeadingNumberPrefix": "Remove prefix numbers",
    "stripLeadingDashes": "Remove leading dashes",
    "removeText": "Remove text",
    "removeCharRange": "Remove a range of characters",
    "categoryBundle": "Category Macro",
    "addTextAtBeginning": "Add text at the beginning",
    "addTextAtEnd": "Add text at the end",
    "replaceText": "Replace text",
    "trim": "Trim",
    "titleCase": "Title Case",
    "collapseWhitespace": "Collapse whitespace",
}


class RulesPanel(ctk.CTkFrame):
    def __init__(self, master, theme: dict, on_change, on_apply, **kwargs):
        super().__init__(master, fg_color=theme["panel"], corner_radius=12, **kwargs)
        self.theme = theme
        self.on_change = on_change
        self.on_apply = on_apply
        self.rules: list[Rule] = []
        self._apply_pending = False
        self._build()

    def _tip(self, widget, key: str) -> None:
        bind_tooltip(widget, TIPS[key], self.theme)

    def _build(self) -> None:
        t = self.theme
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            header,
            text="RULES",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=t["text_mute"],
        ).pack(side="left")

        self.apply_btn = ctk.CTkButton(
            header,
            text="Apply",
            width=60,
            height=24,
            fg_color=t["btn"],
            hover_color=t["btn_hover"],
            text_color=t["text"],
            command=self.on_apply,
        )
        self.apply_btn.pack(side="right", padx=(0, 8))
        self._tip(self.apply_btn, "apply_preview")

        clear_btn = ctk.CTkButton(
            header,
            text="Clear",
            width=50,
            height=24,
            fg_color="transparent",
            hover_color=t["panel_2"],
            text_color=t["accent"],
            command=self._clear_rules,
        )
        clear_btn.pack(side="right")
        self._tip(clear_btn, "clear_rules")

        add_row = ctk.CTkFrame(self, fg_color="transparent")
        add_row.pack(fill="x", padx=16, pady=(0, 8))

        self.add_menu = ctk.CTkOptionMenu(
            add_row,
            values=["+ Add rule…"] + [item["label"] for item in RULE_CATALOG],
            command=self._add_rule,
            fg_color=t["control_bg"],
            button_color=t["border"],
            button_hover_color=t["accent"],
            dropdown_fg_color=t["panel_2"],
            text_color=t["text"],
            width=200,
        )
        self.add_menu.set("+ Add rule…")
        self.add_menu.pack(side="left")
        self._tip(self.add_menu, "add_rule")

        self.stack = ctk.CTkScrollableFrame(self, fg_color=t["card"], corner_radius=8)
        self.stack.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    def set_theme(self, theme: dict) -> None:
        self.theme = theme
        self.configure(fg_color=theme["panel"])
        self.stack.configure(fg_color=theme["card"])
        self.add_menu.configure(
            fg_color=theme["control_bg"],
            button_color=theme["border"],
            button_hover_color=theme["accent"],
            dropdown_fg_color=theme["panel_2"],
            text_color=theme["text"],
        )
        self.set_apply_pending(self._apply_pending)
        self._render()

    def set_rules(self, rules: list[Rule]) -> None:
        self.rules = rules
        self._render()

    def get_rules(self) -> list[Rule]:
        return self.rules

    def set_apply_pending(self, pending: bool) -> None:
        self._apply_pending = pending
        t = self.theme
        if pending:
            self.apply_btn.configure(
                state="normal",
                fg_color=t["accent"],
                hover_color=t["accent_hover"],
                text_color="#ffffff",
            )
        else:
            self.apply_btn.configure(
                state="disabled",
                fg_color=t["btn"],
                hover_color=t["btn_hover"],
                text_color=t["text_mute"],
            )

    def _notify(self) -> None:
        self.on_change()

    def _clear_rules(self) -> None:
        self.rules = []
        self._render()
        self._notify()

    def _add_rule(self, label: str) -> None:
        self.add_menu.set("+ Add rule…")
        if label == "+ Add rule…":
            return
        item = next((i for i in RULE_CATALOG if i["label"] == label), None)
        if not item:
            return
        if item["kind"] == "conditionGroup":
            self.rules.append(
                ConditionGroup(
                    conditions=[Condition(field="name", operator="contains", value="")]
                )
            )
        elif item["op"] == "categoryBundle":
            self.rules.append(
                OpRule(
                    op="categoryBundle",
                    params={"categories": [c.to_dict() for c in make_category_rules()]},
                )
            )
        else:
            params = {}
            if item["op"] == "removeText":
                params = {"text": "", "regex": False}
            self.rules.append(OpRule(op=item["op"], params=params))
        self._render()
        self._notify()

    def _render(self) -> None:
        # Move focus off child entries before destroying them (avoids TclError on stale paths).
        try:
            self.stack.focus_set()
        except Exception:
            pass
        for child in self.stack.winfo_children():
            child.destroy()
        for index, rule in enumerate(self.rules, start=1):
            if isinstance(rule, ConditionGroup):
                self._render_condition_group(rule, index)
            elif isinstance(rule, OpRule):
                self._render_op_rule(rule, index, self.stack, depth=0)

    def _card_title(self, rule: Rule, index: int) -> str:
        if isinstance(rule, ConditionGroup):
            cond = rule.conditions[0] if rule.conditions else Condition()
            val = cond.value or "…"
            return f"{index:02d} · If filename contains '{val}' ({len(rule.children)})"
        if isinstance(rule, OpRule):
            return f"{index:02d} · {OP_LABELS.get(rule.op, rule.op)}"
        return f"{index:02d}"

    def _render_condition_group(self, group: ConditionGroup, index: int) -> None:
        t = self.theme
        card = ctk.CTkFrame(
            self.stack, fg_color=t["input"], corner_radius=10, border_width=1, border_color=t["border_soft"]
        )
        card.pack(fill="x", pady=6, padx=4)

        title_row = ctk.CTkFrame(card, fg_color="transparent")
        title_row.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(
            title_row,
            text=self._card_title(group, index),
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=t["text"],
        ).pack(side="left")

        remove_btn = ctk.CTkButton(
            title_row,
            text="✕",
            width=28,
            height=24,
            fg_color="transparent",
            hover_color=t["border"],
            text_color=t["text_mute"],
            command=lambda g=group: self._remove_rule(g),
        )
        remove_btn.pack(side="right")
        self._tip(remove_btn, "remove_rule")

        cond_row = ctk.CTkFrame(card, fg_color="transparent")
        cond_row.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(cond_row, text="IF", text_color=t["text_mute"], width=24).pack(side="left")
        field_menu = ctk.CTkOptionMenu(
            cond_row, values=["filename"], width=90, fg_color=t["card"], button_color=t["border"],
            text_color=t["text"],
        )
        field_menu.pack(side="left", padx=4)
        self._tip(field_menu, "condition_field")
        op_menu = ctk.CTkOptionMenu(
            cond_row,
            values=["contains", "equals", "matches", "not contains"],
            width=110,
            fg_color=t["card"],
            button_color=t["border"],
            text_color=t["text"],
            command=lambda v, g=group: self._update_condition_op(g, v),
        )
        if group.conditions:
            op_menu.set(group.conditions[0].operator if group.conditions[0].operator != "notContains" else "not contains")
        op_menu.pack(side="left", padx=4)
        self._tip(op_menu, "condition_op")

        value_var = ctk.StringVar(value=group.conditions[0].value if group.conditions else "")
        value_entry = ctk.CTkEntry(
            cond_row,
            textvariable=value_var,
            width=140,
            fg_color=t["input"],
            border_color=t["border"],
            text_color=t["text"],
        )
        value_entry.pack(side="left", padx=4)
        value_entry.bind("<KeyRelease>", lambda e, g=group, v=value_var: self._update_condition_value(g, v))
        value_entry.bind("<FocusOut>", lambda e, g=group, v=value_var: self._update_condition_value(g, v))
        self._tip(value_entry, "condition_value")

        ctk.CTkLabel(cond_row, text="THEN APPLY", text_color=t["text_mute"], font=ctk.CTkFont(size=11)).pack(
            anchor="w", padx=12, pady=(4, 0)
        )

        children_frame = ctk.CTkFrame(card, fg_color="transparent")
        children_frame.pack(fill="x", padx=8, pady=(0, 10))

        for child_index, child in enumerate(group.children, start=1):
            if isinstance(child, OpRule):
                self._render_op_rule(child, child_index, children_frame, depth=1, parent_group=group)

        add_child = ctk.CTkOptionMenu(
            children_frame,
            values=["+ Add child rule…"] + [OP_LABELS.get(i["op"], i["label"]) for i in RULE_CATALOG if i.get("op")],
            command=lambda label, g=group: self._add_child_rule(g, label),
            fg_color=t["card"],
            button_color=t["border"],
            width=180,
            text_color=t["text_dim"],
        )
        add_child.set("+ Add child rule…")
        add_child.pack(anchor="w", padx=12, pady=(4, 0))
        self._tip(add_child, "add_child_rule")

    def _render_op_rule(
        self,
        rule: OpRule,
        index: int,
        parent: ctk.CTkFrame,
        depth: int = 0,
        parent_group: ConditionGroup | None = None,
    ) -> None:
        t = self.theme
        row = ctk.CTkFrame(parent, fg_color=t["card"] if depth else t["input"], corner_radius=8)
        row.pack(fill="x", padx=12 + depth * 8, pady=2)

        enabled_var = ctk.BooleanVar(value=rule.enabled)

        def toggle() -> None:
            rule.enabled = enabled_var.get()
            self._notify()

        enable_sw = ctk.CTkSwitch(
            row,
            text="",
            width=40,
            variable=enabled_var,
            command=toggle,
            progress_color=t["accent"],
            button_color=t["text_mute"],
            button_hover_color=t["text"],
        )
        enable_sw.pack(side="left", padx=(8, 4))
        self._tip(enable_sw, "rule_enable")

        label = OP_LABELS.get(rule.op, rule.op)
        if rule.op == "removeText" and rule.params.get("text"):
            label = f"Remove text ({rule.params['text']})"

        ctk.CTkLabel(
            row,
            text=f"{index:02d}  {label}",
            font=ctk.CTkFont(size=12),
            text_color=t["text"],
        ).pack(side="left", padx=4)

        if rule.op == "removeText":
            text_var = ctk.StringVar(value=rule.params.get("text", ""))
            entry = ctk.CTkEntry(
                row, textvariable=text_var, width=100, fg_color=t["input"], border_color=t["border"]
            )
            entry.pack(side="right", padx=8, pady=4)
            entry.bind(
                "<KeyRelease>",
                lambda e, r=rule, v=text_var: self._update_op_param(r, "text", v.get()),
            )
            self._tip(entry, "remove_text")

        if rule.op == "categoryBundle":
            self._render_category_table(rule, parent)

        if parent_group is None:
            del_btn = ctk.CTkButton(
                row,
                text="✕",
                width=28,
                height=24,
                fg_color="transparent",
                hover_color=t["border"],
                text_color=t["text_mute"],
                command=lambda r=rule: self._remove_rule(r),
            )
            del_btn.pack(side="right", padx=4)
            self._tip(del_btn, "remove_rule")

    def _render_category_table(self, rule: OpRule, parent: ctk.CTkFrame) -> None:
        t = self.theme
        cats_data = rule.params.get("categories", [])
        categories = [CategoryRule.from_dict(c) for c in cats_data]

        table = ctk.CTkFrame(parent, fg_color=t["panel_2"], corner_radius=8)
        table.pack(fill="x", padx=24, pady=(4, 8))

        head = ctk.CTkFrame(table, fg_color="transparent")
        head.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(head, text="PREFIX", font=ctk.CTkFont(size=10, weight="bold"), text_color=t["text_mute"]).pack(
            side="left", padx=(88, 80)
        )
        ctk.CTkLabel(
            head, text="KEYWORDS (COMMA-SEPARATED)", font=ctk.CTkFont(size=10, weight="bold"), text_color=t["text_mute"]
        ).pack(side="left")

        for idx, cat in enumerate(categories):
            row_bg = t["card"]
            if t.get("row_odd"):
                row_bg = t["row_even"] if idx % 2 == 0 else t["row_odd"]
            row = ctk.CTkFrame(table, fg_color=row_bg, corner_radius=4)
            row.pack(fill="x", padx=4, pady=1)

            prefix_bg = category_color(cat.name, cat.color, override=cat.color_override)

            badge = ctk.CTkButton(
                row,
                text=category_badge_label(cat.name),
                width=72,
                height=24,
                fg_color=prefix_bg,
                hover_color=prefix_bg,
                text_color=CATEGORY_BADGE_TEXT,
                corner_radius=4,
                font=ctk.CTkFont(size=10, weight="bold"),
                command=lambda i=idx, r=rule: self._open_color_picker(r, i),
            )
            badge.pack(side="left", padx=(4, 6))
            self._tip(badge, "category_color")
            try:
                badge.configure(cursor="hand2")
            except Exception:
                pass

            prefix_var = ctk.StringVar(value=cat.affix)
            kw_var = ctk.StringVar(value=cat.keywords)

            prefix_entry = ctk.CTkEntry(
                row, textvariable=prefix_var, width=100, fg_color=t["input"], border_color=t["border"]
            )
            prefix_entry.pack(side="left", padx=(0, 8))
            self._tip(prefix_entry, "prefix_field")
            kw_entry = ctk.CTkEntry(
                row, textvariable=kw_var, fg_color=t["input"], border_color=t["border"]
            )
            kw_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self._tip(kw_entry, "keywords_field")

            def save_cat(i=idx, p=prefix_var, k=kw_var, r=rule) -> None:
                cats = r.params.get("categories", [])
                if i < len(cats):
                    cats[i]["affix"] = p.get()
                    cats[i]["keywords"] = k.get()
                    self._notify()

            prefix_entry.bind("<KeyRelease>", lambda e, s=save_cat: s())
            kw_entry.bind("<KeyRelease>", lambda e, s=save_cat: s())
            prefix_entry.bind("<FocusOut>", lambda e, s=save_cat: s())
            kw_entry.bind("<FocusOut>", lambda e, s=save_cat: s())

            del_cat_btn = ctk.CTkButton(
                row,
                text="✕",
                width=24,
                height=24,
                fg_color="transparent",
                hover_color=t["border"],
                text_color=t["text_mute"],
                command=lambda i=idx, r=rule: self._remove_category_row(r, i),
            )
            del_cat_btn.pack(side="right")
            self._tip(del_cat_btn, "remove_category_row")

    def _open_color_picker(self, rule: OpRule, index: int) -> None:
        t = self.theme
        popup = ctk.CTkToplevel(self)
        popup.title("Pick category color")
        popup.resizable(False, False)
        popup.attributes("-topmost", True)
        popup.configure(fg_color=t["panel_2"])

        ctk.CTkLabel(
            popup,
            text="Category colors",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=t["text"],
        ).pack(padx=12, pady=(12, 8))

        grid = ctk.CTkFrame(popup, fg_color="transparent")
        grid.pack(padx=12, pady=(0, 12))

        def choose(color: str) -> None:
            cats = [CategoryRule.from_dict(c) for c in rule.params.get("categories", [])]
            if index < len(cats):
                cats[index].color = color
                cats[index].color_override = True
                rule.params["categories"] = [c.to_dict() for c in cats]
                self._render()
                self._notify()
            popup.destroy()

        cols = 8
        for i, color in enumerate(CATEGORY_PALETTE_COLORS):
            btn = ctk.CTkButton(
                grid,
                text="",
                width=28,
                height=28,
                corner_radius=6,
                fg_color=color,
                hover_color=color,
                border_width=1,
                border_color=t["border"],
                command=lambda c=color: choose(c),
            )
            btn.grid(row=i // cols, column=i % cols, padx=3, pady=3)

        popup.update_idletasks()
        x = self.winfo_rootx() + 40
        y = self.winfo_rooty() + 120
        popup.geometry(f"+{x}+{y}")

    def _remove_category_row(self, rule: OpRule, index: int) -> None:
        cats = rule.params.get("categories", [])
        if 0 <= index < len(cats):
            cats.pop(index)
            rule.params["categories"] = cats
            self._render()
            self._notify()

    def _update_condition_op(self, group: ConditionGroup, op: str) -> None:
        mapped = "notContains" if op == "not contains" else op
        if group.conditions:
            group.conditions[0].operator = mapped
        self._render()
        self._notify()

    def _update_condition_value(self, group: ConditionGroup, var: ctk.StringVar) -> None:
        if group.conditions:
            group.conditions[0].value = var.get()
        self._notify()

    def _update_op_param(self, rule: OpRule, key: str, value: str) -> None:
        rule.params[key] = value
        self._notify()

    def _remove_rule(self, rule: Rule) -> None:
        if rule in self.rules:
            self.rules.remove(rule)
        else:
            for group in self.rules:
                if isinstance(group, ConditionGroup) and rule in group.children:
                    group.children.remove(rule)
                    break
        self._render()
        self._notify()

    def _add_child_rule(self, group: ConditionGroup, label: str) -> None:
        if label == "+ Add child rule…":
            return
        op = next((k for k, v in OP_LABELS.items() if v == label), None)
        if not op:
            item = next((i for i in RULE_CATALOG if i.get("label") == label), None)
            op = item.get("op") if item else None
        if not op:
            return
        params: dict = {}
        if op == "removeText":
            params = {"text": "KSHMR", "regex": False}
        if op == "categoryBundle":
            params = {"categories": [c.to_dict() for c in make_category_rules()]}
        group.children.append(OpRule(op=op, params=params))
        self._render()
        self._notify()
