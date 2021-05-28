Biconvex relaxation for Semidefinite Programming
=================================================

Rodrigue Rillardon, Axel Marchand

Introduction
------------

Dans le cadre du projet de compressed sensing, nous avons
cherché à reproduire les résultats de `cet article <https://arxiv.org/pdf/1605.09527.pdf/>`_.



Installation
------------
Notre code repose sur les librairies suivantes

   - Normalized Cut,     https://github.com/panji1990/Ncut9
   - VLFeat,             https://www.vlfeat.org/install-matlab.html
   - L-bfgs-b,           https://github.com/stephenbeckr/L-BFGS-B-C
   - Min/Max selection,  http://www.mathworks.com.au/matlabcentral/fileexchange/23576-minmax-selection

Il faudra dans un premier temps les installer avant de pouvoir executer les différents programmes.



Organisation
------------

Le fichier SDCut reprends les travaux `l'article suivant <https://arxiv.org/pdf/1304.0840.pdf/>`_. Il est réadapté pour fonctionner
avec la dernière version disponible de MATLAB 2019_B

Test
------------

On peut lancer le fichier d'exemple de segmentation en lançant la commande suivante:

::

        >>> run demo_imgsegm_biased.m


Rapport
-------

Notre rapport est disponible dans le pdf `compressed_sensing_BCR.pdf`.