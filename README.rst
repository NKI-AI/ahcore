AI for Oncology Core for Computational Pathology
================================================

.. image:: https://github.com/NKI-AI/ahcore/actions/workflows/precommit_checks.yml/badge.svg
   :target: https://github.com/NKI-AI/ahcore/actions/workflows/precommit_checks.yml

.. image:: https://codecov.io/gh/NKI-AI/ahcore/branch/main/graph/badge.svg?token=OIJ7F9G7OO
   :target: https://codecov.io/gh/NKI-AI/ahcore

Ahcore are the `AI for Oncology <https://aiforoncology.nl>`_ core components for computational pathology. It provides a set of tools for working with pathology images and annotations. It also offers standard computational pathology algorithms.

Check the `full documentation <https://docs.aiforoncology.nl/ahcore>`_ for more details on how to use ahcore.

License and Usage
-----------------

Ahcore is not intended for clinical use. It is licensed under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

Included Models
---------------

Ahcore comes with several models included, check the `model zoo <https://docs.aiforoncology.nl/ahcore/model_zoo.html>`_ for more information:

- Tissue/background segmentation in H&Es

Quickstart
----------
To train a model, first make sure ahcore is installed:

``git clone https://github.com/NKI-AI/ahcore.git && cd ahcore && pip install -e .``

Next, make sure that the environmental variables are set correctly. We are using ``dotenv`` for this
but you can set the variables themselves. Check ``tools/.env.example`` to see which variables you need to set.
If you use ``dotenv``, you will need to make a copy ``cp tools/.env.example tools/.env`` and fill in the correct values.

Next, you will need to create manifest, and once done, it's often beneficial to copy the data to a local drive, if its
not there yet. You can use the ``ahcore data copy-data-from-manifest`` tool for this.
