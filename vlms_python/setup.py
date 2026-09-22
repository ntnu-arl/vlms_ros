from setuptools import find_packages, setup

package_name = "vlms_python"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Albert Gassol Puigjaner",
    maintainer_email="albert.g.puigjaner@ntnu.no",
    description="VLMs Package",
    license="BSD3",
    entry_points={
        "console_scripts": [],
    },
)
