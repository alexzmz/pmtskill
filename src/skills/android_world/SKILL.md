---
name: android-world-t3a
description: Reliable text-only Android UI operation strategies for Android World tasks.
---

# Android World T3A playbook

Apply this playbook only when choosing or reviewing Android UI actions. Preserve
the response format requested by the surrounding prompt.

## Core policy

1. Parse the goal into an exact target state before acting. Preserve names,
   phone numbers, message bodies, dates, times, durations, units, recurrence,
   and capitalization exactly as requested.
2. Inspect the visible UI and history before every action. Never invent an
   element index. Prefer a visible element whose text, content description, or
   role directly matches the target.
3. Use `open_app` for the intended app instead of navigating through the app
   drawer. Use the shortest reversible path.
4. Perform one action at a time. After every action, use the changed UI and the
   step summary to verify whether it worked.
5. Declare completion only after visible evidence shows the requested final
   state. Reaching the right screen is not the same as completing the task.

## Forms and data entry

- Fill fields in the visual order that minimizes navigation.
- Before typing, verify the target is editable and that its label matches the
  requested value. `input_text` handles focusing and replacing field content;
  do not type through the on-screen keyboard.
- For pickers, tabs, dropdowns, checkboxes, and switches, inspect the current
  value first. Do not toggle a setting that already has the desired state.
- Date and time pickers may open on a default value. Check year, month, day,
  hour, minute, AM/PM, timezone implications, and recurrence separately.
- After filling a form, look for the actual commit control such as Save, Add,
  Create, Send, Done, OK, or a check mark. Handle confirmation dialogs.

## Common app workflows

- Settings: navigate by visible category and verify the final switch or value.
  Search inside Settings only when direct navigation is unclear.
- Contacts: create or edit the contact, commit it, then verify the saved detail
  view contains the requested fields.
- Messages: verify recipient and exact body before Send; after sending, confirm
  the outgoing message appears in the conversation.
- Calendar: verify title plus absolute date/time and recurrence, save, then
  confirm the event appears on the intended date.
- Clock: distinguish alarm, timer, and stopwatch tabs. For alarms, verify
  AM/PM and enabled state. For timers, enter the requested duration before
  starting.
- Notes and tasks: preserve the requested title/body or task text, save through
  the app's actual commit path, and verify the item appears in its list/detail.
- Files and browser tasks: read the full filename and extension. When Android
  shows an app chooser or permission prompt, choose only the option needed for
  the requested flow.

## Recovery

- If an action has no visible effect, check whether a dialog, keyboard,
  permission prompt, loading state, or off-screen control is blocking progress.
- Retry an identical click at most once when the UI plausibly missed it.
  Otherwise use the visible alternative or navigate back one level.
- If the target is not visible, scroll one direction deliberately and compare
  the new elements with the previous screen. Reverse direction if content did
  not move as expected.
- Do not repeat a failed action sequence from history without changing the
  approach.

## Output discipline

- For an action-selection request, emit exactly one valid Android World action
  after a concise `Reason:` and `Action:` label.
- For a step-summarization request, emit only a short, factual one-line summary
  of what changed, whether the action worked, and the best next step. Do not
  emit a new action.
