import os

package_path = os.path.dirname(__file__)
module_path = os.path.join(package_path, 'maven_iuvs/')
docs_path = os.path.join(package_path, 'docs')
os.system('rm -r ' + docs_path)
os.system('pdoc --html --force --output-dir ' + docs_path + ' ' + module_path)
pdoc_path = os.path.join(docs_path, 'maven_iuvs')
os.system('mv -f ' + os.path.join(pdoc_path, '*') + ' ' + docs_path)
os.system('rm -r ' + pdoc_path)
