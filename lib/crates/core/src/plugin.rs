//! The plugin system: named components, typed configs, generated schemas.
//!
//! Every swap point in the library -- chunker, reader, embedder, store,
//! reranker -- is a trait with a [`Registry`] in front of it. A plugin declares
//! a `Config` type; `schemars` derives its JSON Schema; the registry can then
//! build one from untyped JSON and hand back a trait object.
//!
//! That buys three things at once:
//!
//! * **config-driven pipelines** -- a YAML file names `"recursive"` and passes
//!   `{"size": 512}`, with no code change and no `if/elif` dispatch chain;
//! * **validation before work starts** -- a typo in a config is a clear error
//!   at load time, not a surprise forty minutes into an indexing run;
//! * **a machine-readable catalogue** -- [`Registry::describe`] emits every
//!   plugin with its schema, which is what generates Python stubs and docs.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::error::{Error, Result};

/// A reference to a plugin by name, with its configuration.
///
/// Lets one component embed another without knowing which implementation it
/// will get: a BM25 index names its tokenizer, a pipeline names its embedder.
/// Deserialises from either `"simple"` or `{"name": "simple", "config": {...}}`,
/// because most of the time there is nothing to configure.
#[derive(Debug, Clone, serde::Deserialize, JsonSchema)]
#[serde(from = "SpecRepr")]
pub struct PluginSpec {
    pub name: String,
    pub config: serde_json::Value,
}

#[derive(serde::Deserialize, JsonSchema)]
#[serde(untagged)]
enum SpecRepr {
    Named(String),
    Full {
        name: String,
        #[serde(default)]
        config: serde_json::Value,
    },
}

impl From<SpecRepr> for PluginSpec {
    fn from(r: SpecRepr) -> Self {
        match r {
            SpecRepr::Named(name) => Self { name, config: serde_json::Value::Null },
            SpecRepr::Full { name, config } => Self { name, config },
        }
    }
}

impl PluginSpec {
    pub fn named(name: impl Into<String>) -> Self {
        Self { name: name.into(), config: serde_json::Value::Null }
    }
}

/// A constructible, named component with a typed configuration.
pub trait Component: Sized + Send + Sync + 'static {
    type Config: DeserializeOwned + JsonSchema + Default;

    /// The name used in configs. Stable; renaming breaks user configs.
    const NAME: &'static str;
    /// One line, shown in the catalogue.
    const SUMMARY: &'static str;

    fn build(config: Self::Config) -> Result<Self>;
}

type Builder<T> = Box<dyn Fn(&serde_json::Value) -> Result<Box<T>> + Send + Sync>;

struct Entry<T: ?Sized + 'static> {
    build: Builder<T>,
    schema: fn() -> serde_json::Value,
    summary: &'static str,
}

fn schema_of<C: Component>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(C::Config)).unwrap_or(serde_json::Value::Null)
}

/// A named set of interchangeable implementations of one trait.
pub struct Registry<T: ?Sized + 'static> {
    kind: &'static str,
    entries: BTreeMap<&'static str, Entry<T>>,
}

impl<T: ?Sized + 'static> Registry<T> {
    pub fn new(kind: &'static str) -> Self {
        Self { kind, entries: BTreeMap::new() }
    }

    /// Register `C`. `into` performs the unsizing coercion to the trait
    /// object, which the caller supplies as `|c| Box::new(c)`.
    pub fn register<C: Component>(&mut self, into: fn(C) -> Box<T>) -> Result<&mut Self> {
        if self.entries.contains_key(C::NAME) {
            return Err(Error::DuplicatePlugin { kind: self.kind, name: C::NAME.into() });
        }
        let build: Builder<T> = Box::new(move |v: &serde_json::Value| {
            // A missing or null config means "all defaults", so the common case
            // needs no config block at all.
            let cfg: C::Config = if v.is_null() {
                C::Config::default()
            } else {
                serde_json::from_value(v.clone())
                    .map_err(|e| Error::config(C::NAME, e.to_string()))?
            };
            Ok(into(C::build(cfg)?))
        });
        self.entries
            .insert(C::NAME, Entry { build, schema: schema_of::<C>, summary: C::SUMMARY });
        Ok(self)
    }

    pub fn build(&self, name: &str, config: &serde_json::Value) -> Result<Box<T>> {
        match self.entries.get(name) {
            Some(e) => (e.build)(config),
            None => Err(Error::UnknownPlugin {
                kind: self.kind,
                name: name.into(),
                available: self.names().join(", "),
            }),
        }
    }

    /// Build from all defaults.
    pub fn default_build(&self, name: &str) -> Result<Box<T>> {
        self.build(name, &serde_json::Value::Null)
    }

    /// Build from a [`PluginSpec`], as embedded in another component's config.
    pub fn build_spec(&self, spec: &PluginSpec) -> Result<Box<T>> {
        self.build(&spec.name, &spec.config)
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn schema(&self, name: &str) -> Result<serde_json::Value> {
        self.entries.get(name).map(|e| (e.schema)()).ok_or_else(|| Error::UnknownPlugin {
            kind: self.kind,
            name: name.into(),
            available: self.names().join(", "),
        })
    }

    /// The whole catalogue: name, summary and config schema for each plugin.
    pub fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": self.kind,
            "plugins": self.entries.iter().map(|(name, e)| {
                serde_json::json!({
                    "name": name,
                    "summary": e.summary,
                    "config_schema": (e.schema)(),
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl<T: ?Sized + 'static> std::fmt::Debug for Registry<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("kind", &self.kind)
            .field("plugins", &self.names())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    trait Greeter: Send + Sync {
        fn greet(&self) -> String;
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct HelloConfig {
        #[serde(default = "default_name")]
        name: String,
        #[serde(default)]
        excited: bool,
    }
    fn default_name() -> String {
        "world".into()
    }
    impl Default for HelloConfig {
        fn default() -> Self {
            Self { name: default_name(), excited: false }
        }
    }

    struct Hello(HelloConfig);
    impl Component for Hello {
        type Config = HelloConfig;
        const NAME: &'static str = "hello";
        const SUMMARY: &'static str = "Greets by name.";
        fn build(config: Self::Config) -> Result<Self> {
            Ok(Self(config))
        }
    }
    impl Greeter for Hello {
        fn greet(&self) -> String {
            format!("hello {}{}", self.0.name, if self.0.excited { "!" } else { "" })
        }
    }

    fn registry() -> Registry<dyn Greeter> {
        let mut r = Registry::<dyn Greeter>::new("greeter");
        r.register::<Hello>(|c| Box::new(c)).unwrap();
        r
    }

    #[test]
    fn builds_from_defaults_and_from_config() {
        let r = registry();
        assert_eq!(r.default_build("hello").unwrap().greet(), "hello world");
        let g = r.build("hello", &serde_json::json!({"name": "ada", "excited": true})).unwrap();
        assert_eq!(g.greet(), "hello ada!");
    }

    #[test]
    fn an_unknown_plugin_names_what_is_available() {
        let err = registry().default_build("nope").err().unwrap();
        let msg = err.to_string();
        assert!(msg.contains("nope") && msg.contains("hello"), "unhelpful: {msg}");
    }

    #[test]
    fn a_bad_config_fails_at_build_time_not_at_run_time() {
        // deny_unknown_fields turns a typo into an immediate, named error
        // rather than a silently ignored setting.
        let err = registry().build("hello", &serde_json::json!({"nmae": "ada"})).err().unwrap();
        assert!(matches!(err, Error::Config { .. }), "{err}");
        assert!(err.to_string().contains("hello"));
    }

    #[test]
    fn registering_the_same_name_twice_is_rejected() {
        let mut r = registry();
        assert!(matches!(
            r.register::<Hello>(|c| Box::new(c)),
            Err(Error::DuplicatePlugin { .. })
        ));
    }

    #[test]
    fn a_spec_accepts_both_a_bare_name_and_a_full_object() {
        let bare: PluginSpec = serde_json::from_value(serde_json::json!("hello")).unwrap();
        assert_eq!(bare.name, "hello");
        assert!(bare.config.is_null());

        let full: PluginSpec =
            serde_json::from_value(serde_json::json!({"name": "hello", "config": {"name": "ada"}}))
                .unwrap();
        assert_eq!(full.config["name"], "ada");
        assert_eq!(registry().build_spec(&full).unwrap().greet(), "hello ada");
    }

    #[test]
    fn describe_emits_a_usable_schema() {
        let d = registry().describe();
        let p = &d["plugins"][0];
        assert_eq!(p["name"], "hello");
        assert_eq!(p["summary"], "Greets by name.");
        let props = &p["config_schema"]["properties"];
        assert!(props.get("name").is_some(), "schema lacks fields: {props:?}");
        assert!(props.get("excited").is_some());
    }
}
