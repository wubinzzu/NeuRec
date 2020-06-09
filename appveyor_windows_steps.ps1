# $workspace=$clone_folder
# Set-PSDebug -Trace 1

$pyvers = @("36", "36-x64", "37", "37-x64", "38", "38-x64")

foreach ($ver in $pyvers)
{
    $python = "c:\Python${ver}\python"
    & $python -m pip install cython
    & $python -m pip install numpy==1.18.1

    if ($python.contains("-x64"))
    {
        $vc="64"
    }
    else
    {
        $vc="32"
    }

    "C:\Program Files (x86)\Microsoft Visual Studio\2015\Community\VC\Auxiliary\Build\vcvars${vc}.bat"

    & $python setup.py build_ext --inplace clean --all
    pwd
    md ..\compiled_pyd
    xcopy .\*.pyd ..\compiled_pyd /s /y
}
