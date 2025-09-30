{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   {%- if fullname == "bblean" %}
   :no-members:
   {%- endif %}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      :nosignatures:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes and fullname != "bblean" %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :nosignatures:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

{%- block modules %}
{%- if modules %}
{%- if fullname == "bblean" %}
.. toctree::
    :hidden:
    :caption: API reference

    self
{%- endif %}
.. rubric:: Modules

.. autosummary::
   :nosignatures:
   :toctree:
   :recursive:
   :template: custom-module.rst
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
