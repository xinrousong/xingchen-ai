@echo off
REM One-click: train the 300M model (306.5M params) on this laptop.
REM Resume is automatic (checkpoints-base-300m\latest.pt). Watchdog + panel start too.
call "%~dp0start_train.cmd" configs\base-300m.yaml checkpoints-base-300m
