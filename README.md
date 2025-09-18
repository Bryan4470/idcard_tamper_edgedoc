EdgeDoc: Hybrid CNN-Transformer Model for Accurate Forgery Detection and Localization in ID Documents
====================================================================================================

Setup Instructions
------------------

1. **Set up environment**::

      pip install -r requirements

2. **Download and install the FantasyID dataset**:

   `FantasyID Dataset <https://www.idiap.ch/en/scientific-research/data/fantasyid>`_

   Place it in a folder named ``FANTASY``.

3. **Extract and save the ground truth masks**::

      python src/gt_extract.py


Training
--------

After preparing the dataset and masks, launch the training.  
Make sure to adapt the file locations correctly.

Example command::

    python train.py \
      --train-csv FANTASY/FantasyIDiap-ICCV25-Challenge/fantasyIDiap-train.csv \
      --val-csv   FANTASY/FantasyIDiap-ICCV25-Challenge/fantasyIDiap-test.csv \
      --base-dir  FANTASY/FantasyIDiap-ICCV25-Challenge \
      --trufor-dir FANTASY/FantasyIDiap-ICCV25-Challenge/TRUFOROUTPUT \
      --mask-dir   FANTASY/FantasyIDiap-ICCV25-Challenge/GTMASKS \
      --epochs 25 --batch 1 --device cuda --workers 4 --pin-memory \
      --out-dir checkpoints


Testing
-------

Once the model is trained, test it on one image with::

    python src/test_onimg.py
