@echo off

::
:: run this script from the project directory
::

:: generate translation file (.pot) for all .cpp and .h files in the project
for %%A IN (py) do (
	for /R "." %%f in (*.%%A) do (
		xgettext --directory=torchsat --sort-output --keyword=_ --join-existing --output-dir=./translation -o torchsat_imc.pot %%f
	)
)

:: update existing translation (.po file)
cd translation
msgmerge ru.po torchsat_imc.pot --update --sort-output
