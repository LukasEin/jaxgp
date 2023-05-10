import setuptools

setuptools.setup(
    name="jaxgp",
    version="0.0.1",
    author="Lukas Einramhof",
    author_email="lukas.einramhof@gmail.com",
    description=
    "Gaussian Process Regression framework to recover functions from their gradient observations",
    url="https://github.com/LukasEin/jaxgp.git",
    packages=setuptools.find_packages(),
    install_requires=[
        'jax', 'jaxopt', 'matplotlib', 'jupyter', "dropbox"
    ],
    extras_require={'testing': ['pytest>=5.0']},
    python_requires='>=3.7',
    # scripts=['scripts/clustering.py',
    #          'scripts/frame_select.py',
    #          'scripts/gen_soap_descriptors.py',
    #          'scripts/kernel_density_estimation.py',
    #          'scripts/kpca.py',
    #          'scripts/kpca_sparse.py',
    #          'scripts/krr.py',
    #          'scripts/pca.py',
    #          'scripts/kpca_for_projection_viewer.py',
    #          'scripts/ridge_regression.py',
    #          'scripts/tsne.py',
    #          'scripts/umap_reducer.py']
    )