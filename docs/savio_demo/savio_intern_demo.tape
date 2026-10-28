# Savio quick start demo for new lab members (follows pytorchAudio/examples/hubert/savio/INTERN.md).
# Re-shoot after edits:   cd ~/Desktop/vocalizations_lab/docs/savio_demo && PATH=/opt/homebrew/bin:$PATH vhs savio_intern_demo.tape
# (Homebrew ffmpeg must come before the Anaconda one on PATH, which lacks libx264 and fails the MP4 render.)
# Needs a live Savio certificate (`sv login`), because it logs in for real and runs real commands.
# Nothing here changes anything on Savio: install.sh is safe to re-run (it finds the existing env), the
# smoke-test submission is shown as a comment, and every other command only reads.

Output savio_intern_demo.mp4
Output savio_intern_demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1500
Set Height 900
Set Padding 24
Set TypingSpeed 30ms
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set CursorBlink false

Hide
Type "export PS1='laptop$ ' && alias ssh='ssh -o LogLevel=ERROR' && clear"
Enter
Sleep 500ms
Show

Type "# Savio quick start for new lab members -- savio/INTERN.md, step by step"
Enter
Sleep 1.5s
Type "# Step 1. Log in. You type:  ssh YOUR_USERNAME@hpc.brc.berkeley.edu   then your PIN + one-time code."
Enter
Type "# (This recording uses an ssh shortcut with a 12-hour certificate, so no code prompt appears.)"
Enter
Sleep 1.5s
Type "ssh savio-login"
Enter
Wait+Line@90s /\$ ?$/
Sleep 1.5s

Type "# Step 2. Get the code (once). It must land at ~/pytorchAudio:"
Enter
Type "#   git clone -b updated-hyperparams https://github.com/theunissenlab/SpectrogramBasedBERT.git ~/pytorchAudio"
Enter
Type "ls ~/pytorchAudio/examples/hubert/train.py      # worked if this prints the path"
Enter
Wait+Line@60s /\$ ?$/
Sleep 2s

Type "# Step 3. Install: the lab command + hubert_env, the exact Python every model was trained in."
Enter
Type "# The first time this takes 10-30 min; re-running it is safe and quick."
Enter
Type "bash ~/pytorchAudio/examples/hubert/savio/install.sh --env --jupyter"
Enter
Wait+Line@300s /\$ ?$/
Sleep 3s
Type "clear"
Enter
Type "# Then open a NEW terminal. Worked if lab data lists the corpora, each marked yes:"
Enter
Type "lab data | head -26"
Enter
Wait+Line@60s /\$ ?$/
Sleep 5s
Type "lab data combined_it1      # one path, for scripts:  DATA=$(lab data combined_it1)"
Enter
Wait+Line@60s /\$ ?$/
Sleep 2.5s
Type "clear"
Enter

Type "# Step 4. The test job, from the code folder:"
Enter
Type "cd ~/pytorchAudio/examples/hubert"
Enter
Type "#   lab train slurm/smoke_test.sh     -> prints a JOBID (not submitted here: the free queue is busy)"
Enter
Type "#   lab log JOBID -f                  -> follows it; Ctrl-C stops following, not the job"
Enter
Type "# A finished test job ends like this:"
Enter
Type "lab log 39813913 | tail -10"
Enter
Wait+Line@60s /\$ ?$/
Sleep 6s
Type "clear"
Enter

Type "# Step 5. Everyday: what is queued or running, and what ended in the last 3 days?"
Enter
Type "lab status"
Enter
Wait+Line@60s /\$ ?$/
Sleep 6s
Type "clear"
Enter

Type "# Step 7. Scores. The shared leaderboard, then one model's full report:"
Enter
Type "lab eval list"
Enter
Wait+Line@60s /\$ ?$/
Sleep 3s
Type "lab eval show E23_dapt_seed2 | head -22"
Enter
Wait+Line@60s /\$ ?$/
Sleep 9s
Type "clear"
Enter

Type "# Everything else -- a GPU shell, Jupyter, embeddings for your own audio -- is in lab help:"
Enter
Type "lab help | head -44"
Enter
Wait+Line@60s /\$ ?$/
Sleep 10s
Type "exit"
Enter
Sleep 1.5s
