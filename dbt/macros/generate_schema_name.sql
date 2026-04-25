{# Use the configured schema name as the real Postgres schema (e.g. "analytics"), not
   target_schema + "_" + name (which would be public_analytics). #}
{% macro generate_schema_name(custom_schema_name, node) -%}
  {%- if custom_schema_name is none -%}
    {{ target.schema }}
  {%- else -%}
    {{ custom_schema_name | trim }}
  {%- endif -%}
{%- endmacro %}
