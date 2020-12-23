import os

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',))
module_path = os.path.join(package_path, 'maven_iuvs')
os.system('pdoc --html {} --force'.format(module_path))
